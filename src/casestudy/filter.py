import re
import warnings
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, SaltRemover, FilterCatalog

warnings.filterwarnings("ignore", category=RuntimeWarning, message="to-Python converter.*")

params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH)
FILTER_CATALOG = FilterCatalog.FilterCatalog(params)
SALT_REMOVER = SaltRemover.SaltRemover()

ALLOWED_ELEMENTS = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53} # H, B, C, N, O, F, P, S, Cl, Br, I
ATOM_PATTERN = re.compile(r'^(?:[HBCNOFPSI]|Cl|Br|[#=()\[\]+-]|@+|/|\\|[0-9])+$', re.IGNORECASE)

def process_molecule(data_tuple):
    smiles, mol_id, prop_list = data_tuple
    mw, hba, hbd, rot_bonds, fsp3, tpsa, logp = prop_list

    # Initialize Molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (mol_id, None, None, False)

    # Formal Charge Check
    net_charge = Chem.GetFormalCharge(mol)
    if abs(net_charge) > 1:
        return (mol_id, None, None, False)

    # Fragment Handling (Salt Stripping)
    if '.' in smiles:
        initial_atoms = mol.GetNumAtoms()
        mol = SALT_REMOVER.StripMol(mol)
        if mol.GetNumAtoms() < initial_atoms:
            try:
                Chem.SanitizeMol(mol)
            except:
                return (mol_id, None, None, False)

    # Ring & Aromaticity Analysis
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    if ring_count < 1:
        return (mol_id, None, None, False)

    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    if aromatic_rings > 4:
        return (mol_id, None, None, False)

    # Element Validation (Strict)
    atom_elements = {a.GetAtomicNum() for a in mol.GetAtoms()}
    if not atom_elements.issubset(ALLOWED_ELEMENTS):
        return (mol_id, None, None, False)

    # Stereochemical Complexity
    total_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    if total_stereo > 4:
        return (mol_id, None, None, False)
    
    undefined_stereo = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
    if undefined_stereo > 1:
        return (mol_id, None, None, False)

    # Structural Alert Matching
    if FILTER_CATALOG.HasMatch(mol):
        return (mol_id, None, None, False)

    # Canonicalization
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    
    csv_line = f"{mol_id},{canonical_smiles},{mw:.2f},{hbd},{hba},{rot_bonds},{net_charge},{ring_count},{fsp3:.3f},{tpsa:.2f},{aromatic_rings},{total_stereo},{undefined_stereo},{logp:.2f}\n"
    
    return (mol_id, canonical_smiles, csv_line, True)
