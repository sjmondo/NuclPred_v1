#!/usr/bin/env python
###created by sjmondo on 2024-11-05 09:35:34.510622###
"""
generate_matrix.py --assembly in.fasta --data_dir path_to_DNAshapeR_results_folder --outfile results.tab --organism_scaffold_label label
Take assembly and shapes information generated by DNAshapeR (https://www.bioconductor.org/packages/release/bioc/html/DNAshapeR.html) and produce input file for DL prediction.
"""
import sys
sys.dont_write_bytecode = True
import os, argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,usage='generate_matrix.py --assembly in.fasta --data_dir path_to_DNAshapeR_results_folder --outfile results.tab --organism_scaffold_label label',description='Take assembly and shapes information generated by DNAshapeR (https://www.bioconductor.org/packages/release/bioc/html/DNAshapeR.html) and produce input file for DL prediction.')
parser.add_argument('-a', '--assembly', default=None, help='input assembly, fasta format.')
parser.add_argument('-d', '--data_dir', default=None, help='directory containing DNAshapeR results.')
parser.add_argument('-ol', '--organism_scaffold_label', default=None, help='label to attach to each scaffold to denote organism. This is needed for DL prediction later, which can be run on a merged set of sequences from multiple organisms.')

if len(sys.argv) < 2:
    parser.print_help()
    exit()

from Bio import SeqIO
import numpy as np


def shapes_to_dict(shape_file):
    """Take shape information and convert to dict, structured like so; scaffold : [val_pos1, val_pos2 ... val_posN]"""
    shape_dict = {}
    chrom = ''
    for line in open(shape_file):
        if '>' in line:
            if chrom != '' and (shape_file.endswith('.HelT') or shape_file.endswith('.Roll')):
                shape_dict[chrom].append('NA')
            chrom = line.strip('>').split()[0]
            shape_dict[chrom] = []
        else:
            data = line.split(',')
            for datum in data:
                shape_dict[chrom].append(datum.strip())
    if shape_file.endswith('.HelT') or shape_file.endswith('.Roll'):
        shape_dict[chrom].append('NA')
    return shape_dict

def encode_GC(dict_of_sequences):
    """Convert each position into GC or AT, 0 for GC, 1 for AT"""
    converted = {}
    invalid = {}
    for scaff in dict_of_sequences:
        converted[scaff] = []
        for ind, nucleotide in enumerate(dict_of_sequences[scaff]):
            base = dict_of_sequences[scaff][ind].upper()
            if base == 'G' or base == 'C':
                number = 0
            elif base == 'A' or base =='T':
                number = 1
            else:
                if base not in invalid:
                    invalid[base] = 1
                else:
                    invalid[base] += 1
            converted[scaff].append(number)
    if invalid:
        sys.stderr.write('Invalid_bases\tcount:\n')
        for base in invalid:
            sys.stderr.write('%s\t%s\n' % (base, invalid[base]))
    else:
        sys.stderr.write('no invalid bases in the assembly.\n')
    return converted

def convert_assembly(dict_of_sequences, dict_of_identifiers):
    """convert sequences into integers, for downstream processing."""
    converted = {}
    invalid = {}
    for scaff in dict_of_sequences:
        converted[scaff] = []
        for ind, nucleotide in enumerate(dict_of_sequences[scaff]):
            bases = dict_of_sequences[scaff][ind].upper()
            number = dict_of_identifiers[bases]
            if 'N' in bases:
                if bases not in invalid:
                    invalid[bases] = 1
                else:
                    invalid[bases] += 1
            converted[scaff].append(number)
    if invalid:
        sys.stderr.write('Invalid_bases\tcount:\n')
        for base in invalid:
            sys.stderr.write('%s\t%s\n' % (base, invalid[base]))
    else:
        sys.stderr.write('no invalid bases in the assembly.\n')
    return converted

def encode_GC_chunks(encoded_GC_dict):
    """take GC data and generate pct GC 75bp surrounding each site"""
    res = {}
    for scaffold in encoded_GC_dict:
        res[scaffold] = []
        for ind, entry in enumerate(encoded_GC_dict[scaffold]):
            if ind >= 75 and ind < len(encoded_GC_dict[scaffold])-75:
                data_window = encoded_GC_dict[scaffold][ind-75:ind+75+1]
                res[scaffold].append(sum(data_window)/float(len(data_window)))
            else:
                res[scaffold].append(0)
    return res


def make_nuc_combinations_map(kmer_len):
    """take a list comprising of A, T, C, G, N and make all combinations of the specified length"""
    import itertools
    nucleotides = ['A', 'T', 'C', 'G', 'N']
    combos = itertools.product(nucleotides, repeat=kmer_len)
    res = {}
    for ind, combo in enumerate(combos):
        res[''.join(combo)] = ind
    return res

if __name__ == "__main__":
    ARGS = parser.parse_args()
    ASSEMBLY = ARGS.assembly
    DATADIR = ARGS.data_dir
    DBID = ARGS.organism_scaffold_label
    ASSEMB_DICT = {}
    for scaff in SeqIO.parse(open(ASSEMBLY), 'fasta'):
        ASSEMB_DICT[scaff.id] = str(scaff.seq)
    INT_ENCODED_GC = encode_GC(ASSEMB_DICT)
    MONO_MAP = make_nuc_combinations_map(1)
    INT_ENCODED_SCAFFS = convert_assembly(ASSEMB_DICT, MONO_MAP)
    GC_CHUNKS = encode_GC_chunks(INT_ENCODED_GC)
    for infile in os.listdir(DATADIR):
        if infile.endswith('.EP'):
            sys.stderr.write('collecting EP data\n')
            ep = shapes_to_dict('%s/%s' % (DATADIR, infile))
        elif infile.endswith('.Roll'):
            sys.stderr.write('collecting Roll data\n')
            roll = shapes_to_dict('%s/%s' % (DATADIR, infile))
        elif infile.endswith('.ProT'):
            sys.stderr.write('collecting ProT data\n')
            prot = shapes_to_dict('%s/%s' % (DATADIR, infile))
        elif infile.endswith('.HelT'):
            sys.stderr.write('collecting HelT data\n')
            helt = shapes_to_dict('%s/%s' % (DATADIR, infile))
        elif infile.endswith('.MGW'):
            sys.stderr.write('collecting MGW data\n')
            mgw = shapes_to_dict('%s/%s' % (DATADIR, infile))
    sys.stderr.write('writing results to stdout\n')
    SKIPPED = []
    for scaffold in INT_ENCODED_SCAFFS:
        for ind, base in enumerate(INT_ENCODED_SCAFFS[scaffold]):
            print('%s.%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (DBID, scaffold, INT_ENCODED_SCAFFS[scaffold][ind], round(GC_CHUNKS[scaffold][ind], 3), ep[scaffold][ind], helt[scaffold][ind], prot[scaffold][ind], roll[scaffold][ind], mgw[scaffold][ind]))
    for item in SKIPPED:
        sys.stderr.write('%s\n' % (' '.join(str(x) for x in item)))