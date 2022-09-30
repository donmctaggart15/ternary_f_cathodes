# -*- coding: utf-8 -*-


from simmate.database import connect
from simmate.database.third_parties import MatprojStructure
import pandas as pd
import copy
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.cost import CostAnalyzer, CostDBCSV

mpr = MPRester("m9EeydTFK7QrDSZN")       # User's legacy MP API key, new MP API key does not work for interface reactions

#%%   MAJOR SECTIONS OF CODE

#   1: Define Materials class to aid in calculations
#   2: Filtering MatProj database via Simmate 
#   3: Filtering complete F/de-F pairs
#   4: Filtering partial F/de-F pairs
#   5: Calculation of additional F/de-F properties
#   6: Creation of summary table

#%% Create all lists: CAREFUL! WILL ERASE PREVIOUSLY FILTERED DATA 

mpids_list_f = []
mpids_list_def = []
energy_per_atom_list_f = []
energy_per_atom_list_def = []
spacegroup_list_f =[]
spacegroup_list_def =[]
e_hull_list_f =[]
e_hull_list_def =[]
bandgap_list_f =[]
bandgap_list_def =[]
def_type = []

#### 1. Define Materials class to make calculations easier

class Material:
    def __init__(self,mpid):        
        # Initialize an object of this class. The ID is required to initialize the object
        self.structure = MatprojStructure.objects.filter(id=mpid).to_toolkit()[0]
        self.id = mpid
        self.reduced_formula = self.structure.composition.reduced_formula
        self.volume = self.structure.volume
        self.atoms = self.structure.num_sites
        self.formula_units = self.structure.composition.get_reduced_composition_and_factor()[1]
        self.atoms_per_fu = self.atoms / self.formula_units
        self.volume_per_fu = self.volume / self.formula_units
        self.density = self.structure.density 
        self.elements = self.structure.composition.elements
        self.molar_mass = self.structure.composition.weight / self.formula_units
 
# Create scaling factor for each structure pair
def ScaleFactorDeFbyF(m1,m2):
    # Get each structure
    f_struc = MatprojStructure.objects.filter(id=m1.id).to_toolkit()[0]
    def_struc = MatprojStructure.objects.filter(id=m2.id).to_toolkit()[0]
    # Data dict contains list of elements in structure, need to get first element
    data_dict_f = f_struc.composition.to_data_dict
    data_dict_def = def_struc.composition.to_data_dict
    # Reduced dict contains number of each element in reduced form, need number of first element
    reduced_dict_f = f_struc.composition.to_reduced_dict
    reduced_dict_def = def_struc.composition.to_reduced_dict
    # Get the identity of first element in structure
    first_element_f = data_dict_f['elements'][0]
    first_element_def = data_dict_def['elements'][0]
    # Some strucs have F as first element, if so, use the following element
    if first_element_f == 'F':
        first_element_f = data_dict_f['elements'][1]
    elif first_element_def == 'F':
        first_element_def = data_dict_def['elements'][1]
    # Get the number of the first element in reduced form
    num_first_element_f = reduced_dict_f[first_element_f]
    num_first_element_def = reduced_dict_def[first_element_def]
    # Divide number of first element for deF by number of first element for F
    scale_factor= num_first_element_def / num_first_element_f    
    return scale_factor
    
# Number of F atoms transferred between pair
def FTransfer(m1,m2,scale_factor):
    # Some F-transfers were incorrect due to differences in formula units per unit cell
    if scale_factor != 1:       
        new_m2 = m2.atoms_per_fu / scale_factor
        return m1.atoms_per_fu - new_m2
    else:
        return m1.atoms_per_fu - m2.atoms_per_fu

# Volume per mobile F relative to fluorinated structure
def VolumePerF(m1,atoms_lost):
    return m1.volume_per_fu / atoms_lost

# Percent expansion from defluorinated to fluorinated
def VolumeChange(m1, m2, scale_factor):
    if scale_factor != 1:
        return ((m1.volume_per_fu / (m2.volume_per_fu / scale_factor)) - 1) * 100
    else:    
        return ((m1.volume_per_fu / m2.volume_per_fu) - 1) * 100

# Voltage of F/deF pair relative to Li/Li+
def Voltage(m1_f, m2_def, e_per_a_f, e_per_a_def, f_transfer, scale_factor):
    # -9.6903 is LiF energy and -1.909 is Li energy
    if scale_factor != 1:
        return (((m2_def.atoms_per_fu / scale_factor)* e_per_a_def + f_transfer * -9.6903) - (m1_f.atoms_per_fu * e_per_a_f + f_transfer * -1.909)) / -f_transfer
    else:
        return ((m2_def.atoms_per_fu * e_per_a_def + f_transfer * -9.6903) - (m1_f.atoms_per_fu * e_per_a_f + f_transfer * -1.909)) / -f_transfer         

# Gravimetric capacity (mAh/g) of fluorinated structure
def GravCapacity(f_transfer, mw):
    return ((f_transfer * 96485.3321) / (3.6 * mw )) # mAh/g = Ah/kg

# Gravimetric energy density (Wh/kg) of fluorinated structure
def GravEnDen(grav_capacity, voltage):  # Ah/kg * V = Wh/kg
    return grav_capacity * voltage 

# Volumetric energy density (Wh/L) of fluorinated structure
def VolEnDen(grav_eden, density):
    return grav_eden * density 

#### 2.1: FILTER MATPROJ DATABASE TO INCLUDE ALL F STRUCTURES WITH MORE THAN 2 ELEMENTS WITHIN 75meV OF HULL

print("\nBegin MatProj database filtering (all F-structures)")

# Initial filter, order by hull energy so that duplicates can be deleted correctly
all_f_strucs = MatprojStructure.objects.filter(
    elements__icontains='"F"',
    energy_above_hull__lt=0.075,
    nelements__gt=2,
).order_by('energy_above_hull').all()

# Make dataframe
df_all_f_strucs = all_f_strucs.to_dataframe()

# Keep only lowest energy structure, first occurance will be lowest energy (ordering), delete all others
df_filter_f_strucs = df_all_f_strucs.drop_duplicates("formula_reduced")
df_filter_f_strucs = df_filter_f_strucs.reset_index(drop=True)

# Make list of ids to make Structure objects
ids = df_filter_f_strucs['id'].tolist()

# Make copies of tables for each filtering type
strucs_df_complete = copy.copy(df_filter_f_strucs)
strucs_df_partial = copy.copy(df_filter_f_strucs)

#### 2.2: Make list of Structure objects from ids so that F can be removed

list_filter_f_strucs = []

# Make list of Structure objects
# Note: this makes list of lists so entries must be called with x[0] (not just x) 
for i,x in enumerate(ids):
    struc = MatprojStructure.objects.filter(
        id=x
    ).to_toolkit()
    list_filter_f_strucs.append(struc)
    
# Make copies for each filtering type
strucs_list_complete = copy.copy(list_filter_f_strucs)
strucs_list_partial = copy.copy(list_filter_f_strucs)

print("\nMatProj database filtering complete:", len(list_filter_f_strucs),"F-containing structures")

#### 3.1: COMPLETE DEF PAIRS FILTER

print("\nBegin complete de-F filtering")

for i,struc in enumerate(strucs_list_complete):
    
    # Get id, energy per atom, spacegroup, and hull energy for F structure
    mpid_f = strucs_df_complete.id[i]
    energy_per_atom_f = strucs_df_complete.energy_per_atom[i]
    spacegroup_f = strucs_df_complete.spacegroup[i]
    e_hull_f = strucs_df_complete.energy_above_hull[i]
    bandgap_f = strucs_df_complete.band_gap[i]
        
    # Remove F from each entry in data_mp_list
    struc_deF = copy.copy(struc[0])
    struc_deF.remove_species("F")
    
    # For each entry check if there is another MP entry with reduced formula (formula_reduced)
    # identical to the reduced fromula after removing F (struc.composition.reduced_formula)
    possible_all = MatprojStructure.objects.filter(
        formula_reduced=struc_deF.composition.reduced_formula,  
        energy_above_hull__lte=0.075,
    ).order_by('energy_above_hull').all()
   
    # Make dataframe
    possible_df = possible_all.to_dataframe()
    
    # Keep only lowest energy structure, first occurance will be lowest energy (ordering), delete all others
    filter_possible_df = possible_df.drop_duplicates("formula_reduced")
    filter_possible_df = filter_possible_df.reset_index(drop=True)
    
    # If deF structure exists, get its info and add F/deF info to property lists
    if len(filter_possible_df) > 0:
        print("Complete de-F match for", strucs_df_complete.formula_reduced[i], "-->", filter_possible_df.formula_reduced[0])
        mpid_def = filter_possible_df.id[0]
        energy_per_atom_def = filter_possible_df.energy_per_atom[0]
        spacegroup_def = filter_possible_df.spacegroup[0]
        e_hull_def = filter_possible_df.energy_above_hull[0]
        bandgap_def = filter_possible_df.band_gap[0]
     
        mpids_list_f.append(mpid_f)
        mpids_list_def.append(mpid_def)
        energy_per_atom_list_f.append(energy_per_atom_f)
        energy_per_atom_list_def.append(energy_per_atom_def)
        spacegroup_list_f.append(spacegroup_f)
        spacegroup_list_def.append(spacegroup_def)
        e_hull_list_f.append(e_hull_f)
        e_hull_list_def.append(e_hull_def)
        bandgap_list_f.append(bandgap_f)
        bandgap_list_def.append(bandgap_def)
        def_type.append("Complete")   
    else:
        print("No complete de-F match for index:", i)    
    
#### 3.2: Use this to copy lists so that complete filter process doesnt have to run again

mpids_list_f_copy = copy.copy(mpids_list_f)
mpids_list_def_copy = copy.copy(mpids_list_def) 

energy_per_atom_list_f_copy = copy.copy(energy_per_atom_list_f) 
energy_per_atom_list_def_copy = copy.copy(energy_per_atom_list_def) 
spacegroup_list_f_copy =copy.copy(spacegroup_list_f) 
spacegroup_list_def_copy =copy.copy(spacegroup_list_def) 
e_hull_list_f_copy =copy.copy(e_hull_list_f) 
e_hull_list_def_copy = copy.copy(e_hull_list_def) 
def_type_copy = copy.copy(def_type)  
bandgap_list_f_copy = copy.copy(bandgap_list_f)  
bandgap_list_def_copy = copy.copy(bandgap_list_def)   

#### 3.3: Restore from copied lists if problem occured

# mpids_list_f = copy.copy(mpids_list_f_copy)
# mpids_list_def = copy.copy(mpids_list_def_copy) 

# energy_per_atom_list_f = copy.copy(energy_per_atom_list_f_copy) 
# energy_per_atom_list_def = copy.copy(energy_per_atom_list_def_copy) 
# spacegroup_list_f =copy.copy(spacegroup_list_f_copy) 
# spacegroup_list_def =copy.copy(spacegroup_list_def_copy) 
# e_hull_list_f =copy.copy(e_hull_list_f_copy) 
# e_hull_list_def = copy.copy(e_hull_list_def_copy) 
# def_type = copy.copy(def_type_copy) 
#bandgap_list_f = copy.copy(bandgap_list_f_copy)  
#bandgap_list_def = copy.copy(bandgap_list_def_copy)   

#### 4.1 PARTIAL DEF PAIRS FILTER 

print("\nBegin partial de-F filtering")

for i,struc in enumerate(strucs_list_partial):
    
    mpid2_f = strucs_df_partial.id[i]
    energy_per_atom2_f = strucs_df_partial.energy_per_atom[i]
    spacegroup2_f = strucs_df_partial.spacegroup[i]
    e_hull2_f = strucs_df_partial.energy_above_hull[i]
    bandgap2_f = strucs_df_partial.band_gap[i]
    
    # This filtering only catches structures that have the same chemical system, but not same ratio of atoms
    possible_all = MatprojStructure.objects.filter(  # NOTE: THIS EXCLUDES ALL FULLY DEFLUORINATED STRUCTURES
        elements=strucs_df_partial.elements[i],  # Check that partial deF entries have same chemical system as original
        energy_above_hull__lte=0.075,
        nelements__gt=2,
    ).exclude(nsites=struc[0].composition.num_atoms).order_by('energy_above_hull').all()
    
    # Make dataframe
    possible_df = possible_all.to_dataframe()
    
    # Keep only lowest energy structure, first occurance will be lowest energy (ordering), delete all others
    filter_possible_df = possible_df.drop_duplicates("formula_reduced")
    filter_possible_df = filter_possible_df.reset_index(drop=True)
   
    # Make list of structures so that F species can be removed 
    ids = filter_possible_df['id'].tolist()
    filter_possible_list = []
    for j,x in enumerate(ids):
        struc2 = MatprojStructure.objects.filter(
            id=x
        ).to_toolkit()
        filter_possible_list.append(struc2)
   
    #print(i, len(possible_df), len(filter_possible_df), len(filter_possible_list))
   
    if len(filter_possible_df) > 0:
        all_elements = struc[0].composition.elements
        struc_def = copy.copy(struc[0])
        struc_def.remove_species("F")
        
        for k,m in enumerate(filter_possible_list):
            possible_def = copy.copy(m[0])
            possible_def.remove_species("F")
            
            struc_def_element1 = struc_def.composition.get_atomic_fraction(all_elements[0])
            struc_def_element2 = struc_def.composition.get_atomic_fraction(all_elements[1])
            struc_def_element3 = struc_def.composition.get_atomic_fraction(all_elements[2])
            possible_def_element1 = possible_def.composition.get_atomic_fraction(all_elements[0])
            possible_def_element2 = possible_def.composition.get_atomic_fraction(all_elements[1])
            possible_def_element3 = possible_def.composition.get_atomic_fraction(all_elements[2])
            struc_elementF = struc[0].composition.get_atomic_fraction(all_elements[-1])
            possible_elementF = m[0].composition.get_atomic_fraction(all_elements[-1])
        
            if struc_def_element1 == possible_def_element1 and struc_def_element2 == possible_def_element2 and struc_def_element3 == possible_def_element3 and struc_elementF > possible_elementF:                   
                print("Match for partial de-F, same elements and ratio:", struc[0].composition.reduced_formula, "-->", m[0].composition.reduced_formula)               
                mpid2_def = filter_possible_df.id[k]
                energy_per_atom2_def = filter_possible_df.energy_per_atom[k]
                spacegroup2_def = filter_possible_df.spacegroup[k]
                e_hull2_def = filter_possible_df.energy_above_hull[k]
                bandgap2_def = filter_possible_df.band_gap[k]
              
                mpids_list_f.append(mpid2_f)
                mpids_list_def.append(mpid2_def)
                energy_per_atom_list_f.append(energy_per_atom2_f)
                energy_per_atom_list_def.append(energy_per_atom2_def)
                spacegroup_list_f.append(spacegroup2_f)
                spacegroup_list_def.append(spacegroup2_def)
                e_hull_list_f.append(e_hull2_f)
                e_hull_list_def.append(e_hull2_def)
                bandgap_list_f.append(bandgap2_f)
                bandgap_list_def.append(bandgap2_def)
                def_type.append("Partial")             
    else:
        print("No partial de-F matches for index:", i)           
                
#### 4.2: Use this to copy lists so that partial filter process doesnt have to run again

mpids_list_f_copy = copy.copy(mpids_list_f)
mpids_list_def_copy = copy.copy(mpids_list_def)   

energy_per_atom_list_f_copy = copy.copy(energy_per_atom_list_f) 
energy_per_atom_list_def_copy = copy.copy(energy_per_atom_list_def) 
spacegroup_list_f_copy =copy.copy(spacegroup_list_f) 
spacegroup_list_def_copy =copy.copy(spacegroup_list_def) 
e_hull_list_f_copy =copy.copy(e_hull_list_f) 
e_hull_list_def_copy = copy.copy(e_hull_list_def) 
def_type_copy = copy.copy(def_type)  
bandgap_list_f_copy = copy.copy(bandgap_list_f)  
bandgap_list_def_copy = copy.copy(bandgap_list_def)   
    
#### 5.1: Instantiate each ID as a Material object

print("\nBegin Material object creation for each pair")

materials_list_f = []
materials_list_def = []

# Create list of Material objects for F and DeF structures
for i,x in enumerate(mpids_list_f):
    y = mpids_list_def[i]
    materials_list_f.append(Material(x))
    materials_list_def.append(Material(y))
    print("Material creation: Done with index",i,x)
        
#### 5.2: CALCULATE PROPERTIES FOR EACH MATERIAL OBJECT

# Create lists of other properties
reduced_formula_f = []
reduced_formula_def = []
f_transfer_list = []
voltage_list = []
composition_list_f = []
composition_list_def = []
volume_per_f_list = []
volume_change_list = []
density_list_f = []
grav_capacity_list = []
vol_e_density_list = []
grav_e_density_list = []
atoms_per_fu_f =[]
atoms_per_fu_def =[]
molar_mass_list = []
scale_factor_list = []

# Calculate properties for structures
for i,x in enumerate(materials_list_f):
    y = materials_list_def[i]
    j = mpids_list_f[i]
    k = mpids_list_def[i]
    
    scale_factor_list.append(ScaleFactorDeFbyF(x,y))
    sf = scale_factor_list[i]
    
    reduced_formula_f.append(x.reduced_formula)
    reduced_formula_def.append(y.reduced_formula)
    composition_list_f.append(x.structure.composition.reduced_composition)
    composition_list_def.append(y.structure.composition.reduced_composition)
    f_transfer_list.append(FTransfer(x,y,sf))
    print(x.reduced_formula, "-->", y.reduced_formula, FTransfer(x,y,sf))
    voltage_list.append(Voltage(x, y, energy_per_atom_list_f[i], energy_per_atom_list_def[i], f_transfer_list[i], sf))
    volume_per_f_list.append(VolumePerF(x, f_transfer_list[i]))
    volume_change_list.append(VolumeChange(x,y, sf))
    density_list_f.append(x.density)
    atoms_per_fu_f.append(x.atoms_per_fu)
    atoms_per_fu_def.append(y.atoms_per_fu)
    molar_mass_list.append(x.molar_mass)
    grav_capacity_list.append(GravCapacity(f_transfer_list[i], molar_mass_list[i]))
    grav_e_density_list.append(GravEnDen(grav_capacity_list[i], voltage_list[i]))
    vol_e_density_list.append(VolEnDen(grav_e_density_list[i], density_list_f[i]))
    print("Other properties calculated for index:",i)
    
#### 5.3 Calculate energy density for LiCoO2

# MPIDs for LiCoO2, CoO2, graphite (C6), and LiC6
licoo2_id = "mp-22526"
coo2_id = "mp-754748"
carb_id = "mp-569304"
carb_li_id = "mp-1001581"

# Create materials for each Li compound
li_mat = Material(licoo2_id)
coo2_mat = Material(coo2_id)
carb_mat = Material(carb_id)
carb_li_mat = Material(carb_li_id)

# Get energy per atom for each compound
li_epa = MatprojStructure.objects.filter(id=licoo2_id).to_dataframe().energy_per_atom
coo2_epa = MatprojStructure.objects.filter(id=coo2_id).to_dataframe().energy_per_atom
carb_epa = MatprojStructure.objects.filter(id=carb_id).to_dataframe().energy_per_atom
carb_li_epa = MatprojStructure.objects.filter(id=carb_li_id).to_dataframe().energy_per_atom
li_apfu = li_mat.atoms_per_fu
coo2_apfu = coo2_mat.atoms_per_fu
carb_apfu = carb_mat.atoms_per_fu
carb_li_apfu = carb_li_mat.atoms_per_fu

# Calculate energy densities for LiCoO2 (1 Li transferred between LiCoO2 and CoO2)
li_den = li_mat.density
li_voltage = ((4*li_epa[0] + 6*carb_epa[0]) - (3*coo2_epa[0] + 7*carb_li_epa[0])) / -1        
li_grav_cap = GravCapacity(1, li_mat.molar_mass)
li_grav_eden = GravEnDen(li_grav_cap, li_voltage)
li_vol_eden = VolEnDen(li_grav_eden, li_den)

#### 5.4: INTERFACE REACTIONS

print("\nBegin interface reactions")

products_list= []
direct_rxn = []

for i,x in enumerate(reduced_formula_f): 
    
        print("Index:",i, reduced_formula_f[i], "-->", reduced_formula_def[i])
        # Perform interface rxn with F and de-F structure
        prod = mpr.get_interface_reactions(                   
            reduced_formula_f[i], reduced_formula_def[i],  relative_mu=-1, use_hull_energy=False
        )
        print(prod[1])
        # Get the 'rxn' string (reactions with different 'rxn' and 'rxn_str' strings do not have a 'products' entry)
        reaction = prod[1].get('rxn')  
        # Turn rxn string into text so that it can be partitioned into products string           
        reaction_text = f'{reaction}'      
        # Get text string of products after the reaction arrow 
        reaction_products = reaction_text.partition("> ")[2]       
        print("Reaction products", reaction_products, "\n")
        # Add reaction_products to "master" list of products
        products_list.append(reaction_products)      
        # If products equals either reactant (F or de-F), then the reaction is direct (Yes), otherwise, a reaction occured so it is not direct (No)                                     
        for entry in products_list:                 
           if products_list[i] == reduced_formula_f[i] or products_list[i] == reduced_formula_def[i]:
               direct_rxn.append('Yes')
               break
           else:
               direct_rxn.append('No')
               break

### 5.5: COST ANALYSIS
    
print("\nBegin cost analysis")

import math

# Call upon modified cost spreadsheet based on Wikipedia "Price of chemical elements" page
cost_database = CostDBCSV('costdb_elements_new.csv')
cost_analyzer = CostAnalyzer(cost_database)

cost_per_kg_list_f = []
cost_per_mol_list_f = []
cost_per_kg_list_def = []
cost_per_mol_list_def = []
cost_per_mol_per_f_list_f = []
log_cost_p_mol_p_f_list_f = []

for i,x in enumerate(composition_list_f):
    y = composition_list_def[i]
    # Uranium-containing compounds create an error, skip these compounds 
    try:
        cost_per_kg_f = cost_analyzer.get_cost_per_kg(x)
    except:
        cost_per_kg_list_f.append("--")
        cost_per_mol_list_f.append("--")
        cost_per_mol_per_f_list_f.append("--")
        log_cost_p_mol_p_f_list_f.append("--")
        cost_per_kg_list_def.append("--")
        cost_per_mol_list_def.append("--")
        continue
    
    # Get cost/kg and cost/mol for each F and deF
    cost_per_kg_f = cost_analyzer.get_cost_per_kg(x) 
    cost_per_mol_f = cost_analyzer.get_cost_per_mol(x) 
    cost_per_kg_def = cost_analyzer.get_cost_per_kg(y)
    cost_per_mol_def = cost_analyzer.get_cost_per_mol(y)    
    cost_per_mol_per_f_f = cost_per_mol_f / f_transfer_list[i]
    log_cost_p_mol_p_f_f = math.log10(cost_per_mol_per_f_f)
    
    # Add costs to appropriate list
    cost_per_kg_list_f.append(cost_per_kg_f)
    cost_per_mol_list_f.append(cost_per_mol_f)
    cost_per_kg_list_def.append(cost_per_kg_def)
    cost_per_mol_list_def.append(cost_per_mol_def)
    cost_per_mol_per_f_list_f.append(cost_per_mol_per_f_f)
    log_cost_p_mol_p_f_list_f.append(log_cost_p_mol_p_f_f)

# Calculate cost for LIB cathode (LiCoO2)
li_comp = Composition("LiCoO2")
li_cost_p_kg = cost_analyzer.get_cost_per_kg(li_comp)
li_cost_p_mol = cost_analyzer.get_cost_per_mol(li_comp)
log_li_cost_p_mol= math.log10(li_cost_p_mol)

### 5.6: ACTIVATION ENERGY

print("\nCheck activation energies")

# Load activation energy list from reference 23
activation_energy_table = pd.read_csv('F_transport_activation_energies.csv')

activation_ids = activation_energy_table['structure_id'].tolist()
activation_energies = activation_energy_table['approx_barrier_corrected'].tolist()

activation_energy_list_f = []
activation_energy_list_def = []

# Check F structures for activation energy 
for i,x in enumerate(mpids_list_f):
    if x in activation_ids:
        activation_energy_list_f.append(activation_energies[i])
        
    else:
        activation_energy_list_f.append("-")
        
        
# Check def structres for activation energy        
for i,x in enumerate(mpids_list_def):
    if x in activation_ids:
        activation_energy_list_def.append(activation_energies[i])
        
    else:
        activation_energy_list_def.append("-")
        

#### 6.1: CREATE TABLE WITH ALL INFORMATION/PROPERTIES

complete_def_pairs_table = pd.DataFrame({
    'f_form': reduced_formula_f,
    'def_form': reduced_formula_def,
    'voltage': voltage_list,
    'vol_p_f': volume_per_f_list,
    'per_exp': volume_change_list,
    'grav_cap': grav_capacity_list,
    'grav_e_den': grav_e_density_list,
    'vol_e_den': vol_e_density_list,
    'f_transfer': f_transfer_list,
    'f_cost_p_kg': cost_per_kg_list_f,
    'def_cost_p_kg': cost_per_kg_list_def,
    'f_cost_p_mol': cost_per_mol_list_f,
    'def_cost_p_mol': cost_per_mol_list_def,
    'cost_p_mol_f': cost_per_mol_per_f_list_f,
    'log10_cost_p_mol_f': log_cost_p_mol_p_f_list_f,
    'f_id': mpids_list_f,
    'def_id': mpids_list_def,
    'f_spacegroup': spacegroup_list_f,
    'def_spacegroup': spacegroup_list_def,
    'f_e_hull': e_hull_list_f,
    'def_e_hull': e_hull_list_def,
    'f_bandgap': bandgap_list_f,
    'def_bandgap': bandgap_list_def,
    'f_barrier': activation_energy_list_f,
    'def_barrier': activation_energy_list_def,
    'direct': direct_rxn,
    'products': products_list,
    'type': def_type,
    })

complete_def_pairs_table.to_csv('ternary_f_cathodes_all.csv')


print("\nFinal table successfully created")


