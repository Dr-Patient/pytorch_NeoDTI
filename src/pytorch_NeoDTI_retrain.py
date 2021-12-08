import numpy as np 
import torch, os, argparse, random
from sklearn.metrics import roc_auc_score, average_precision_score
basedir = os.path.abspath(os.path.dirname(__file__))
os.chdir(basedir)
os.makedirs("models", exist_ok=True)
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)
from model import NeoDTI
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=26, help="random seed for initialization")
    parser.add_argument("--d", default=1024, type=int, help="the embedding dimension d")
    parser.add_argument("--n",default=1.0, type=float, help="global gradient norm to be clipped")
    parser.add_argument("--k",default=512, type=int, help="the dimension of reprojection matrices k")
    parser.add_argument("--l2-factor",default = 0.1, type=float, help="weight of l2 loss")
    parser.add_argument("--lr", default=1e-3, type=float, help='learning rate')
    parser.add_argument("--weight-decay", default=0, type=float, help='weight decay of the optimizer')
    parser.add_argument("--num-steps", default=3000, type=int, help='number of training steps')
    parser.add_argument("--device", choices=[-1,0,1,2,3], default=0, type=int, help='device number (-1 for cpu)')
    args = parser.parse_args()
    return args

def row_normalize(a_matrix, substract_self_loop):
    if substract_self_loop == True:
        np.fill_diagonal(a_matrix,0)
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1)+1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return torch.Tensor(new_matrix)

def retrain(args, DTItrain, verbose=True):
    set_seed(args)
    drug_protein = np.zeros((num_drug,num_protein))
    mask = np.zeros((num_drug,num_protein))
    for ele in DTItrain:
        drug_protein[ele[0],ele[1]] = ele[2]
        mask[ele[0],ele[1]] = 1
    protein_drug = drug_protein.T

    drug_protein_normalize = row_normalize(drug_protein,False).to(device)
    protein_drug_normalize = row_normalize(protein_drug,False).to(device)
    drug_protein = torch.Tensor(drug_protein).to(device)
    mask = torch.Tensor(mask).to(device)
    model = NeoDTI(args, num_drug, num_disease, num_protein, num_sideeffect)
    model.to(device)
    no_decay = ["bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.8, patience=2)

    ground_truth_train = [ele[2] for ele in DTItrain]
    best_train_aupr = 0
    best_train_auc = 0
    for i in range(args.num_steps):
        model.train()
        model.zero_grad()
        tloss, dtiloss, results = model(drug_drug_normalize, drug_chemical_normalize, drug_disease_normalize, 
                                        drug_sideeffect_normalize, protein_protein_normalize, protein_sequence_normalize, 
                                        protein_disease_normalize, disease_drug_normalize, disease_protein_normalize, 
                                        sideeffect_drug_normalize, drug_protein_normalize, protein_drug_normalize, 
                                        drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, 
                                        protein_sequence, protein_disease, drug_protein, mask)
        # print(results)
        tloss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.n)
        optimizer.step()
        if i % 25 == 0 and verbose == True:
            print('step', i, 'total and dti loss', tloss.item(), dtiloss.item())
            model.eval()
            pred_list_train = [results[ele[0],ele[1]] for ele in DTItrain]
            train_auc = roc_auc_score(ground_truth_train, pred_list_train)
            train_aupr = average_precision_score(ground_truth_train, pred_list_train)
            scheduler.step(train_aupr)
            if train_aupr >= best_train_aupr:
                best_train_aupr = train_aupr
                best_train_auc = train_auc
                torch.save(model, "models/NeoDTI_retrain.pth")
            print ('train auc aupr', train_auc, train_aupr)
    
    return best_train_auc, best_train_aupr


if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    device = torch.device("cuda:{}".format(args.device)) if args.device >= 0 else torch.device("cpu")
    network_path = '../data/'
    print('loading networks ...')
    drug_drug = np.loadtxt(network_path+'mat_drug_drug.txt')
    true_drug = 708 # First [0:708] are drugs, the rest are compounds retrieved from ZINC15 database
    drug_chemical = np.loadtxt(network_path+'Similarity_Matrix_Drugs.txt')
    drug_chemical=drug_chemical[:true_drug,:true_drug]
    drug_disease = np.loadtxt(network_path+'mat_drug_disease.txt')
    drug_sideeffect = np.loadtxt(network_path+'mat_drug_se.txt')
    disease_drug = drug_disease.T
    sideeffect_drug = drug_sideeffect.T

    protein_protein = np.loadtxt(network_path+'mat_protein_protein.txt')
    protein_sequence = np.loadtxt(network_path+'Similarity_Matrix_Proteins.txt')
    protein_disease = np.loadtxt(network_path+'mat_protein_disease.txt')
    disease_protein = protein_disease.T


    print('normalize network for mean pooling aggregation')
    drug_drug_normalize = row_normalize(drug_drug,True).to(device)
    drug_chemical_normalize = row_normalize(drug_chemical,True).to(device)
    drug_disease_normalize = row_normalize(drug_disease,False).to(device)
    drug_sideeffect_normalize = row_normalize(drug_sideeffect,False).to(device)

    protein_protein_normalize = row_normalize(protein_protein,True).to(device)
    protein_sequence_normalize = row_normalize(protein_sequence,True).to(device)
    protein_disease_normalize = row_normalize(protein_disease,False).to(device)

    disease_drug_normalize = row_normalize(disease_drug,False).to(device)
    disease_protein_normalize = row_normalize(disease_protein,False).to(device)
    sideeffect_drug_normalize = row_normalize(sideeffect_drug,False).to(device)

    #define computation graph
    num_drug = len(drug_drug_normalize)
    num_protein = len(protein_protein_normalize)
    num_disease = len(disease_protein_normalize)
    num_sideeffect = len(sideeffect_drug_normalize)

    drug_drug = torch.Tensor(drug_drug).to(device)
    drug_chemical = torch.Tensor(drug_chemical).to(device)
    drug_disease = torch.Tensor(drug_disease).to(device)
    drug_sideeffect = torch.Tensor(drug_sideeffect).to(device)
    protein_protein = torch.Tensor(protein_protein).to(device)
    protein_sequence = torch.Tensor(protein_sequence).to(device)
    protein_disease = torch.Tensor(protein_disease).to(device)

    # prepare drug_protein and mask
    dti_o = np.loadtxt(network_path+'mat_drug_protein.txt')
    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(dti_o)[0]):
        for j in range(np.shape(dti_o)[1]):
            if int(dti_o[i][j]) == 1:
                whole_positive_index.append([i,j])
            elif int(dti_o[i][j]) == 0:
                whole_negative_index.append([i,j])
    negative_sample_index = np.arange(len(whole_negative_index))
    data_set = np.zeros((len(negative_sample_index)+len(whole_positive_index),3),dtype=int)
    count = 0
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1
    print('Retraining Network')
    retrain(args=args, DTItrain=data_set)