# -*- coding: utf-8 -*-
import csv
import argparse
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np


def get_fingerprint(mol, fingerprint="Morgan"):
    if fingerprint == "Morgan":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_str = DataStructs.cDataStructs.BitVectToText(fp)
    fp_lst = list(fp_str)
    fp_arr = np.array(fp_lst, dtype=int)
    return fp_arr


def format_svmlight(qid, score, fp_arr, comment):
    # fp_arr -> fp_svmlight
    nonzero_index = np.nonzero(fp_arr)[0]
    desc_lst = [str(i + 1) + ":1" for i in nonzero_index]
    ret = [score, "qid:{}".format(qid)] + desc_lst + ["#", comment]
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert PubChem sdf to svmlight")
    parser.add_argument("pubchem_sdf")
    parser.add_argument("datatable")
    parser.add_argument("score_name")
    parser.add_argument("-p", action="store_true",
                        help="if IC50 is specified, please transform to pIC50")
    parser.add_argument("-o", "--out_file", default="./out.svmlight")
    parser.add_argument("-q", "--qid", type=int, default=1)
    args = parser.parse_args()

    temp = []
    with open(args.datatable) as datatable_fp:
        reader = csv.DictReader(datatable_fp)
        for row in reader:
            sid = row["PUBCHEM_SID"]
            score = row[args.score_name]
            if args.p:
                try:
                    score = -np.log10(float(score) * 1e-6)
                except:
                    print("not compound line")
            activity = row["PUBCHEM_ACTIVITY_OUTCOME"]
            temp.append((sid, (score, activity)))
    datatable_dict = OrderedDict(temp)

    input_mols = Chem.SDMolSupplier(args.pubchem_sdf)
    with open(args.out_file, "w") as out_fp:
        writer = csv.writer(out_fp, delimiter=" ")
        N = len(input_mols)
        for i, input_mol in enumerate(input_mols):
            print("{} in {}".format(i + 1, N))
            if input_mol is None:
                print("compound {} is None".format(i))
                continue
            sid = input_mol.GetProp("PUBCHEM_SUBSTANCE_ID")
            score, activity = datatable_dict[sid]
            if score == "":
                continue
            fp_arr = get_fingerprint(input_mol)
            row_svmlight = format_svmlight(args.qid, score, fp_arr, activity)
            writer.writerow(row_svmlight)
