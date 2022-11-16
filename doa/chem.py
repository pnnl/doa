from rdkit import Chem
from rdkit.Chem import Descriptors
import rdkit
from rdkit.Chem import AllChem
import numpy as np
from rdkit.Chem.rdmolops import Get3DDistanceMatrix
from rdkit.Chem import rdMolTransforms
import pandas as pd
from rdkit.Chem.rdmolops import Get3DDistanceMatrix
from tqdm import tqdm
from rdkit.Chem.Draw import rdMolDraw2D
import ast



HDonorSmarts = Chem.MolFromSmarts('[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]')

HAcceptorSmarts = Chem.MolFromSmarts('[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' +
                                     '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' +
                                     '$([nH0,o,s;+0])]')

_HDonors = lambda x, y=HDonorSmarts: x.GetSubstructMatches(y, uniquify=1)
_HAcceptors = lambda x, y=HAcceptorSmarts: x.GetSubstructMatches(y, uniquify=1)



def mark_acceptors(mol, acceptor_idx):
    for atom_idx in acceptor_idx:
        assert len(atom_idx) == 1    
        mol.GetAtomWithIdx(atom_idx[0]).SetBoolProp('IsHbondAcceptor', True)
    # return mol

def mark_donors(mol, donor_idx):
    for atom_idx in donor_idx:
        assert len(atom_idx) == 1    
        mol.GetAtomWithIdx(atom_idx[0]).SetBoolProp('IsHbondDonor', True)
    # return mol

def set_acc_donors(mol):
    mol = Chem.AddHs(mol)
    numatoms = mol.GetNumAtoms()
    
    for i in range(numatoms):
        mol.GetAtomWithIdx(i).SetBoolProp('IsHbondDonor', False)
        mol.GetAtomWithIdx(i).SetBoolProp('IsHbondAcceptor', False)

    donor_idx = _HDonors(mol)
    acceptor_idx = _HAcceptors(mol)
    mark_acceptors(mol, acceptor_idx)
    mark_donors(mol, donor_idx)
    
    return mol




def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def find_intraH_smiles(all_data):
    
    needed_smiles=[]
    for dat_index in tqdm(all_data.index):

        smiles = all_data.smiles[ dat_index ]
        mol_rdk = Chem.MolFromSmiles(smiles)

        if len(Chem.GetMolFrags(mol_rdk)) == 1:


            mol_rdk = set_acc_donors(mol_rdk)
            molwt = Descriptors.ExactMolWt(mol_rdk)

            mol_with_atom_index(mol_rdk)
            num_atoms = mol_rdk.GetNumAtoms()    
            atoms_rd = mol_rdk.GetAtoms()    
            atomic_nums = [a.GetAtomicNum() for a in atoms_rd]


            if (len(set(atomic_nums).difference([1, 6, 7, 8, 9, 16, 17, 35])) == 0) and (molwt>250 and molwt<=800) \
            and (len(set(atomic_nums).intersection([7,8])) != 0):

#             if (len(set(atomic_nums).difference([1, 6, 7, 8, 9, 16, 17, 35])) == 0) and (molwt>250 and molwt<=800) \
#             and ( (7 in atomic_nums) or (8 in atomic_nums)  ):


                needed_smiles.append(smiles)
    return needed_smiles



def get_intraH_mols(PATH_LEN=5, smiles_list=None):
    
    res_intra = []
    
    for smiles in smiles_list:
        mol_rdk = Chem.MolFromSmiles( smiles )
        
        # mol_rdk = set_acc_donors(mol_rdk)
        # add Hydrogen
        # mol_rdk = Chem.AddHs(mol_rdk)

        # create pybel mol
        # mol_pb = pybel.readstring("smi", df.smiles[ j ])
        # add Hydrogen
        # mol_pb.addh()

        # get atoms for rdkit mol
#         mol_rdk = Chem.AddHs(mol_rdk)
        mol_rdk = set_acc_donors(mol_rdk)
#         molwt = Descriptors.ExactMolWt(mol_rdk)

        mol_with_atom_index(mol_rdk)
        num_atoms = mol_rdk.GetNumAtoms()    
        atoms_rd = mol_rdk.GetAtoms()    
        atomic_nums = [a.GetAtomicNum() for a in atoms_rd]

#         if (len(set(atomic_nums).difference([1, 6, 7, 8, 9, 16, 17, 35])) == 0) and (molwt>250 and molwt<=800) \
#         and (7 in atomic_nums) or (8 in atomic_nums):

        Hs = [a for a in atoms_rd if a.GetAtomicNum()==1]
        H_ids = [i.GetIdx() for i in Hs]

#             for atom_id in O_ids:
#                 res = Chem.FindAllPathsOfLengthN(mol=mol_rdk, length=5, useBonds=False, useHs=True, rootedAtAtom=atom_id)

#         all_paths=[]
#         for atom_id in H_ids:
#             res = Chem.FindAllPathsOfLengthN(mol=mol_rdk, length=PATH_LEN, useBonds=False, useHs=True, rootedAtAtom=atom_id)
#             paths = [list(i) for i in res]
#             all_paths.extend(paths)  
        paths_for_abc, paths_for_d = [], []
        for atom_id in H_ids:
            res = Chem.FindAllPathsOfLengthN(mol=mol_rdk, length=PATH_LEN, useBonds=False, useHs=True, rootedAtAtom=atom_id)
            paths = [list(i) for i in res]
            paths_for_abc.extend(paths)

            res = Chem.FindAllPathsOfLengthN(mol=mol_rdk, length=PATH_LEN+1, useBonds=False, useHs=True, rootedAtAtom=atom_id)
            paths = [list(i) for i in res]
            paths_for_d.extend(paths)

        
            


#             paths = [list(i) for i in res]
#             desired_paths = np.where([ mol_rdk.GetAtomWithIdx( p[-2] ).GetAtomicNum() == 7 and \
#                                       mol_rdk.GetAtomWithIdx( p[-1] ).GetAtomicNum() == 1 and \
#                                       for p in all_paths])[0]

#                 OPTION (c)
        desired_paths_c = np.where([ mol_rdk.GetAtomWithIdx( p[1] ).GetAtomicNum() == 7 and \
                                   mol_rdk.GetAtomWithIdx( p[-1] ).IsInRing() and \
                                  mol_rdk.GetAtomWithIdx( p[-1] ).GetAtomicNum() == 7 and \
                                 (mol_rdk.GetAtomWithIdx( p[-2] ).GetAtomicNum() in [6,7])
                                    for p in paths_for_abc])[0]

#                 OPTION (d)
        desired_paths_d = np.where([ mol_rdk.GetAtomWithIdx( p[1] ).GetAtomicNum() == 7 and \
                                  mol_rdk.GetAtomWithIdx( p[-2] ).GetAtomicNum() == 8 and \
                                  mol_rdk.GetAtomWithIdx( p[-1] ).GetAtomicNum() == 6 and \
                                    mol_rdk.GetBondBetweenAtoms(p[-1], p[-2]).GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE
#                                   True in [a.GetAtomicNum() == 6 for a in mol_rdk.GetAtomWithIdx( p[-2] ).GetNeighbors()]
                                    for p in paths_for_d])[0]
#        OPTION (a) and (b)
        desired_paths_ab = np.where([ mol_rdk.GetAtomWithIdx( p[1] ).GetAtomicNum() == 7 and \
                                  mol_rdk.GetAtomWithIdx( p[-1] ).GetAtomicNum() == 8 and \
                                  (mol_rdk.GetAtomWithIdx( p[-2] ).GetAtomicNum() in [6, 16]) and \
                                  mol_rdk.GetAtomWithIdx( p[-1] ).GetBonds()[0].GetBondType() == rdkit.Chem.rdchem.BondType.DOUBLE
                              for p in paths_for_abc])[0]

        desired_paths_ab = desired_paths_ab.tolist() 
        desired_paths_d = desired_paths_d.tolist()
        desired_paths_c = desired_paths_c.tolist()
        
#         path_dict = {'ab': desired_paths_ab, 'c': desired_paths_c, 'd': desired_paths_d}
        path_dict = {'ab': [desired_paths_ab, paths_for_abc],
                 'c': [desired_paths_c, paths_for_abc],
                 'd': [desired_paths_d, paths_for_d]}
        desired_paths = list(set(desired_paths_ab + desired_paths_d + desired_paths_c ))

#     mol_rdk.GetAtomWithIdx( p[0] ).GetBoolProp('IsHbondAcceptor')
#     mol_rdk.GetBondBetweenAtoms(p[0], p[1]).GetBondType() == rdkit.Chem.rdchem.BondType.DOUBLE

        if len(desired_paths) > 0:

            AllChem.EmbedMolecule(mol_rdk)
            try:
                distmat = Get3DDistanceMatrix(mol_rdk)
            except:
                continue
            conf = mol_rdk.GetConformer(0)

            for path_type, dpaths in path_dict.items():
                
                for dp in dpaths[0]:

                    cpath = dpaths[1][dp]
                    dist = distmat[  cpath[0], cpath[-1] ]

                    angle1 = rdMolTransforms.GetAngleDeg(conf,cpath[-1],cpath[0],cpath[1])
    #                     angle2 = rdMolTransforms.GetAngleDeg(conf,curr_path[1],curr_path[2],curr_path[3])
    #                     dhangle = abs(rdMolTransforms.GetDihedralDeg(conf, curr_path[0],curr_path[1],curr_path[2],curr_path[3]))

    #                     print(angle1, angle2, dhangle)
                    c1 = np.logical_and(angle1>= 100, angle1 <= 180)
    #                     c2 = np.logical_and(angle2> 115, angle2 < 125)
    #                     c3 = np.logical_and(dhangle> 0, dhangle < 1.5)


                    if (dist < 2.5) and c1:
    #                     print(dat_index, cpath, dist, cpath[0], cpath[-1])
                        res_intra.append([smiles, cpath, dist, path_type])
    #                         aa = 1
    #                         break

    res_intra = pd.DataFrame(res_intra, columns=['smiles', 'cpath', 'dist', 'path_type'])
    
    return res_intra


def save_images(dft, save_path = None):
    
    for j in dft.index:
        smiles = dft.loc[j, 'smiles']
        path = dft.loc[j, 'path']

        mol = Chem.MolFromSmiles(smiles)
        mol_with_atom_index(mol)
#         path = ast.literal_eval(path)
        mol.__sssAtoms = path
        hit_bonds = [mol.GetBondBetweenAtoms(i[0],i[1]).GetIdx() for i in [(path[i], path[i+1]) for i in range(len(path)-1)]
                    ]
        d = rdMolDraw2D.MolDraw2DCairo(500, 500) # or MolDraw2DCairo to get PNGs MolDraw2DSVG
        # d.drawOptions().addAtomIndices = True
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=path, highlightBonds=hit_bonds)
        d.WriteDrawingText(f'{save_path}/mol{j}.png')


        
        
        
# def find_Zpath(mol_rdk, acc_don_pairs, path_l):

#     found = False
#     all_paths = []
#     for adp in acc_don_pairs:
#         # for each acceptor donor pair,
#         acc = adp[0] # acceptor
#         don = adp[1] # donor

#         # find all the paths of length 6 that starts from acceptor
# #         res = Chem.FindAllPathsOfLengthN(mol=mol_rdk, length=6, useBonds=False,useHs=True, rootedAtAtom=acc)
#         res = Chem.FindAllPathsOfLengthN(mol=mol_rdk, length=path_l, useBonds=False, useHs=True, rootedAtAtom=acc)
#         paths = [list(i) for i in res] # all the paths

#         # check whether donor is the end atom of any of the paths
#         desired_paths = np.where([p[-1] == don for p in paths])[0]


#         # have to do the following for each path

#         for dp in desired_paths:
#             curr_path = paths[ dp ]
# #             print(curr_path)


#             for i in range(len(curr_path)-1):

#                 bond_stereo = mol_rdk.GetBondBetweenAtoms(curr_path[i], curr_path[i+1] ).GetStereo()
# #                 if (bond_stereo == rdkit.Chem.rdchem.BondStereo.STEREOZ) and c1 and c2 and c3:

#                 # if there is a stero bond in the path,
#                 if (bond_stereo == rdkit.Chem.rdchem.BondStereo.STEREOZ):
        
#                     AllChem.EmbedMolecule(mol_rdk)
#                     conf = mol_rdk.GetConformer(0)

#                     angle1 = rdMolTransforms.GetAngleDeg(conf,curr_path[0],curr_path[1],curr_path[2])
#                     angle2 = rdMolTransforms.GetAngleDeg(conf,curr_path[1],curr_path[2],curr_path[3])
#                     dhangle = abs(rdMolTransforms.GetDihedralDeg(conf, curr_path[0],curr_path[1],curr_path[2],curr_path[3]))

#                     print(angle1, angle2, dhangle)
#                     c1 = np.logical_and(angle1> 115, angle1 < 125)
#                     c2 = np.logical_and(angle2> 115, angle2 < 125)
#                     c3 = np.logical_and(dhangle> 0, dhangle < 1.5)
                    
#                     if c1 and c2:

#                         found = True
#                         print("path: ", curr_path)
#                         all_paths.append(curr_path)

    
#     del mol_rdk

#     return found, all_paths



        
# from PIL import Image
# from PIL import Image, ImageDraw, ImageFilter
# from PIL import ImageFont

# n_rows = 10

# files = os.listdir(f"str_data_ols/images/str_ol")
# img = Image.new('RGB', (1800,n_rows*300), (255, 255, 255))
# draw = ImageDraw.Draw(img)

# # for j in range(1,15):
# j1=0
# lbl=1
# old_new = {}
# # for jj in range(0, 21, 3):
# cl = 0

# num_images = res_ol.shape[0]
# if num_images%3 == 0:
#     main_range = num_images
# else:
#     main_range = num_images-3

# for jj in range(0, 10, 3):
#     print("row = ", jj)

#     for jiter in range(6):
# #         j = cluster_keys[jj + jiter]
# #         nmols = len([f for f in files if f.startswith(f"grp{j}_mol")])

# #         for i in range(nmols):

#             im1 = Image.open(f'./str_data_ols/images/str_ol/mol{cl}.png')
#             img.paste(im1,  (jiter*300, j1*300))
#             cl +=1
# #         old_new[j] = cl

#         #            row loc        col loc 
# #         if text_pos[jj + jiter] == 'right':
# #             text_loc = (i*5 + jiter*600 + 450, j1*200+10)
# #         elif text_pos[jj + jiter] == 'center':
# #             text_loc = (i*5 + jiter*600 + 225, j1*200+10)
# #         else:
# #             text_loc = (i*5 + jiter*600, j1*200+10)
# #         draw.text(text_loc, "cluster "+str(cl) ,(0,0,0), font=font)


#     j1+=1