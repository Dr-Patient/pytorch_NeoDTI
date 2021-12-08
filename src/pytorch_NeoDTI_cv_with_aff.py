import numpy as np 
import torch, os, argparse, random
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
basedir = os.path.abspath(os.path.dirname(__file__))
os.chdir(basedir)
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)
from model import NeoDTI_with_aff
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=26, help="random seed for initialization")
    parser.add_argument("--d", default=1024, type=int, help="the embedding dimension d")
    parser.add_argument("--n",default=1.0, type=float, help="global gradient norm to be clipped")
    parser.add_argument("--k",default=512, type=int, help="the dimension of reprojection matrices k")
    parser.add_argument("--l2-factor",default = 0.2, type=float, help="weight of l2 loss")
    parser.add_argument("--lr", default=1e-3, type=float, help='learning rate')
    parser.add_argument("--weight-decay", default=0, type=float, help='weight decay of the optimizer')
    parser.add_argument("--num-steps", default=3000, type=int, help='number of training steps')
    parser.add_argument("--device", choices=[-1,0,1,2,3], default=0, type=int, help='device number (-1 for cpu)')
    parser.add_argument("--n-folds", default=10, type=int, help="number of folds for cross validation")
    parser.add_argument("--round", default=1, type=int, help="number of rounds of sampling")
    parser.add_argument("--test-size", default=0.05, type=float, help="portion of validation data w.r.t. trainval-set")
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

def train_and_evaluate(args, DTItrain, DTIvalid, DTItest, aff, verbose=True):
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

    drop = 0
    for ele in DTItest:
        c1x = ele[0]
        c1y = ele[1]
        if int(ele[2]) == 0:
            continue
        for c2x in range(np.shape(aff)[0]):
            if aff[c2x][c1y] > 0 and drug_chemical[c1x][c2x] > 0.6:
                aff[c2x][c1y] = 0
                drop += 1
    print('drop', drop)
    drug_protein_affinity = aff
    protein_drug_affinity = drug_protein_affinity.T
    drug_protein_affinity_normalize = row_normalize(drug_protein_affinity,False).to(device)
    protein_drug_affinity_normalize = row_normalize(protein_drug_affinity,False).to(device)
    drug_protein_affinity = torch.Tensor(drug_protein_affinity).to(device)

    mask_affinity = np.zeros((num_drug,num_protein))
    mask_affinity[true_drug:,:] = 1
    mask_affinity = torch.Tensor(mask_affinity).to(device)

    model = NeoDTI_with_aff(args, num_drug, num_disease, num_protein, num_sideeffect)
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
    # ground_truth = []  # for evaluation
    # ground_truth_test = []
    ground_truth_train = [ele[2] for ele in DTItrain]
    ground_truth_valid = [ele[2] for ele in DTIvalid]
    ground_truth_test = [ele[2] for ele in DTItest]
    # for ele in DTIvalid:
    #     ground_truth.append(ele[2])
    # for ele in DTItest:
    #     ground_truth_test.append(ele[2])

    best_valid_aupr = 0
    best_valid_auc = 0
    test_aupr = 0
    test_auc = 0
    for i in range(args.num_steps):
        model.train()
        model.zero_grad()
        tloss, dtiloss, results = model(drug_drug_normalize, drug_chemical_normalize, drug_disease_normalize, 
                                        drug_sideeffect_normalize, protein_protein_normalize, protein_sequence_normalize, 
                                        protein_disease_normalize, disease_drug_normalize, disease_protein_normalize, 
                                        sideeffect_drug_normalize, drug_protein_normalize, protein_drug_normalize, 
                                        drug_protein_affinity_normalize, protein_drug_affinity_normalize,
                                        drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, 
                                        protein_sequence, protein_disease, drug_protein, drug_protein_affinity,
                                        mask, drug_drug_mask, drug_disease_mask, drug_sideeffect_mask, mask_affinity)
        # print(results)
        tloss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.n)
        optimizer.step()
        if i % 25 == 0 and verbose == True:
            print('step', i, 'total and dti loss', tloss.item(), dtiloss.item())
            model.eval()
            pred_list_valid = [results[ele[0],ele[1]] for ele in DTIvalid]
            valid_auc = roc_auc_score(ground_truth_valid, pred_list_valid)
            valid_aupr = average_precision_score(ground_truth_valid, pred_list_valid)

            pred_list_train = [results[ele[0],ele[1]] for ele in DTItrain]
            train_auc = roc_auc_score(ground_truth_train, pred_list_train)
            train_aupr = average_precision_score(ground_truth_train, pred_list_train)
            scheduler.step(train_aupr)
            if valid_aupr >= best_valid_aupr:
                best_valid_aupr = valid_aupr
                best_valid_auc = valid_auc
                pred_list_test = [results[ele[0],ele[1]] for ele in DTItest]
                test_auc = roc_auc_score(ground_truth_test, pred_list_test)
                test_aupr = average_precision_score(ground_truth_test, pred_list_test)
            print ('train auc aupr', train_auc, train_aupr, 'valid auc aupr,', valid_auc, valid_aupr, 'test auc aupr', test_auc, test_aupr)
    
    return best_valid_auc, best_valid_aupr, test_auc, test_aupr


if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    device = torch.device("cuda:{}".format(args.device)) if args.device >= 0 else torch.device("cpu")
    network_path = '../data/'
    print('loading networks ...')
    temp = np.loadtxt(network_path+'mat_drug_drug.txt')
    true_drug = 708 # First [0:708] are drugs, the rest are compounds retrieved from ZINC15 database

    drug_chemical = np.loadtxt(network_path+'Similarity_Matrix_Drugs.txt')

    temp = np.loadtxt(network_path+'mat_drug_drug.txt')
    drug_drug = np.zeros((len(drug_chemical),len(drug_chemical)))
    drug_drug[:true_drug,:true_drug] = temp[:,:]


    drug_disease = np.loadtxt(network_path+'mat_drug_disease.txt')
    drug_disease = np.concatenate((drug_disease, np.zeros((len(drug_chemical)-true_drug,np.shape(drug_disease)[1]))), axis=0)

    drug_sideeffect = np.loadtxt(network_path+'mat_drug_se.txt')
    drug_sideeffect = np.concatenate((drug_sideeffect, np.zeros((len(drug_chemical)-true_drug,np.shape(drug_sideeffect)[1]))), axis=0)

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

    #create some masks
    drug_drug_mask = torch.zeros((len(drug_drug),len(drug_drug)), device=device)
    drug_drug_mask[:true_drug,:true_drug] = 1

    drug_disease_mask = torch.zeros((len(drug_drug),np.shape(drug_disease)[1]), device=device)
    drug_disease_mask[:true_drug,:] = 1

    drug_sideeffect_mask = torch.zeros((len(drug_drug),np.shape(drug_sideeffect)[1]), device=device)
    drug_sideeffect_mask[:true_drug,:] = 1

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
    test_auc_round = []
    test_aupr_round = []
    val_auc_round = []
    val_aupr_round = []

    affinity_network = np.loadtxt(network_path+'mat_compound_protein_bindingaffinity.txt')
    dti_o = np.loadtxt(network_path+'mat_drug_protein.txt')
    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(dti_o)[0]):
        for j in range(np.shape(dti_o)[1]):
            if int(dti_o[i][j]) == 1:
                whole_positive_index.append([i,j])
            else:
                whole_negative_index.append([i,j])

    for r in range(args.round):
        print ('sample round', r+1)
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),size=10*len(whole_positive_index),replace=False)

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

        val_auc_fold = []
        val_aupr_fold = []
        test_auc_fold = []
        test_aupr_fold = []
        rs = np.random.randint(0,1000,1)[0]
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=rs)
       
        for train_index, test_index in skf.split(np.arange(len(data_set)), data_set[:,2]):
            DTItrain, DTItest = data_set[train_index], data_set[test_index]
            DTItrain, DTIvalid =  train_test_split(DTItrain, test_size=args.test_size, random_state=rs)

            v_auc, v_aupr, t_auc, t_aupr = train_and_evaluate(args=args, DTItrain=DTItrain, DTIvalid=DTIvalid, DTItest=DTItest, aff=affinity_network)
            val_auc_fold.append(v_auc)
            val_aupr_fold.append(v_aupr)
            test_auc_fold.append(t_auc)
            test_aupr_fold.append(t_aupr)
            # break
        val_auc_round.append(np.mean(val_auc_fold))
        val_aupr_round.append(np.mean(val_aupr_fold))
        test_auc_round.append(np.mean(test_auc_fold))
        test_aupr_round.append(np.mean(test_aupr_fold))
        np.savetxt('val_auc_aff', val_auc_round)
        np.savetxt('val_aupr_aff', val_aupr_round)
        np.savetxt('test_auc_aff', test_auc_round)
        np.savetxt('test_aupr_aff', test_aupr_round)
