# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 09:36:56 2022

@author: donmc
"""

from simmate.shortcuts import setup
from simmate.database.third_parties import (
    MatProjStructure,
    JarvisStructure,
    CodStructure,
    OqmdStructure,
)
import pandas as pd
import copy

from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
#from pymatgen.ext.matproj import MPRester
from mp_api.client import MPRester    # This version needed for "thermo" properties (hull energy, bandgap)
from pymatgen.analysis.cost import CostAnalyzer, CostDBCSV
#from pymatgen.analysis.cost import CostAnalyzer, CostDBElements, CostDBCSV

mpr = MPRester("KCWNf75lwOINtIUANaNCP9Xfdof52mTC")


#   v3 filters out only 1 strucure of given composition
#    Also make seprate data_mp_tables for complete/partial deF
#    Wikipedia prices
#   v4 adds this for initial data_mp_table filtering:
# for j,y in enumerate(indices_of_repeats):
#   multi_hull_e.append(data_hull_energy[y])
# for j,y in enumerate(indices_of_repeats):  
#     if data_hull_energy[y] != min(multi_hull_e) and y not in structure_index_to_delete:   
#         structure_index_to_delete.append(y)
#         structure_rf_to_delete.append(data_reduced_form[y])
#   v5 divides cost per mol by atoms in formula unit
#   v7 adds corrected hull energy column from MP via mpr.thermo (DOES NOT FILTER BASED ON CORRECTED HULL ENERGY)
#   v8 switches to Materials API vs Simmate and filters based on corrected hull energy
#   v9: v8 frogot to do 200mev hull energy for complete/partial filters, also add bandgap


#%%   MAJOR SECTIONS OF CODE

#   1: Filtering MatProj database via Simmate and MP API 
#   2: Filtering complete F/de-F pairs
#   3: Filtering partial F/de-F pairs
#   4: Calculation of additional F/de-F properties
#   5: Creation of summary table


#%% Create all lists: CAREFUL! WILL ERASE DATA 


mpids_list_f = []
mpids_list_def = []
energy_per_atom_list_f = []
energy_per_atom_list_def = []
spacegroup_list_f =[]
spacegroup_list_def =[]
e_hull_list_f =[]
e_hull_list_def =[]
def_type = []


#%% Define Materials class to make calculations easier

class Material:
    def __init__(self,mpid):        # Initialize an object of this class. The ID is required to initialize the object
        self.structure = mpr.get_structure_by_material_id(mpid)
        self.corrected_hull_energy = mpr.thermo.get_data_by_id(mpid).energy_above_hull  
        self.e_per_atom = mpr.thermo.get_data_by_id(mpid).energy_per_atom
        self.reduced_formula = self.structure.composition.reduced_formula
        self.volume = self.structure.volume
        self.atoms = self.structure.num_sites
        self.formula_units = self.structure.composition.get_reduced_composition_and_factor()[1]
        self.atoms_per_fu = self.atoms / self.formula_units
        self.volume_per_fu = self.volume / self.formula_units
        self.density = self.structure.density 
        self.elements = self.structure.composition.elements
        
        
def BandGap(mpid):
        entry = mpr.summary.search(material_ids=[mpid], fields=["band_gap"])  
        if len(entry) > 0:
            return entry[0].band_gap
        else:
            return "--"
    
def FTransfer(m1,m2):
    return m1.atoms_per_fu - m2.atoms_per_fu

def VolumePerF(m1,atoms_lost):
    return m1.volume_per_fu / atoms_lost

def VolumeChange(m1, m2):
    return ((m1.volume_per_fu / m2.volume_per_fu) - 1) * 100

def Voltage(m1_f, m2_def, e_per_a_f, e_per_a_def, f_transfer):
    # -9.6903 is LiF energy and -1.909 is Li energy
    return ((m2_def.atoms_per_fu * e_per_a_def + f_transfer * -9.6903) - (m1_f.atoms_per_fu * e_per_a_f + f_transfer * -1.909)) / -f_transfer         



#%% 1.1: FILTER MATPROJ DATABASE TO INCLUDE ALL F STRUCTURES WITH MORE THAN 2 ELEMENTS WITHIN 200meV OF HULL


data_mp_list = MatProjStructure.objects.filter(
    elements__icontains='"F"',
    # is_stable=True,
    energy_above_hull__lte=0.2,
    nelements__gt="2",
).to_toolkit()

data_mp_table = MatProjStructure.objects.filter(
    elements__icontains='"F"',
    # is_stable=True,
    energy_above_hull__lte=0.2,
    nelements__gt="2",
).to_dataframe()

#%% 1.2: CHECK EACH STRUCTURE AGAINST AN EXISTING MP ENTRY TO SEE IF HULL ENERGY IS ACTUALLY BELOW 75MEV
        # SKIP ENTRIES THAT HAVE BEEN DELETED OR HAVE HULL ENERGY ABOVE 75MEV
        # !!THIS TAKES A LONG TIME TO RUN!! (~20min)
        
        
mev75_reduced_form = []
mev75_hull_energy = []
mev75_index_to_keep = []
mev75_index_to_delete = []
mev75_mpid = []
yes_entry= 0
no_entry = 0
too_big = 0

for i,x in enumerate(data_mp_list):
    mpid = data_mp_table.id[i]
    
    try: 
        w = mpr.get_entry_by_material_id(mpid)
    except:
        print("\nNo entry for",i, mpid)
        no_entry = no_entry + 1
        mev75_index_to_delete.append(i)
        continue
        
    corr_e_hull = mpr.thermo.get_data_by_id(mpid).energy_above_hull    
    if corr_e_hull <= 0.075:
        mev75_reduced_form.append(data_mp_table.formula_reduced[i])
        mev75_hull_energy.append(corr_e_hull)
        mev75_mpid.append(mpid)
        mev75_index_to_keep.append(i)
        yes_entry = yes_entry + 1
        print("\nDone with",i, mpid, corr_e_hull)
    else:
        print("\nToo large for",i)
        too_big = too_big + 1
        mev75_index_to_delete.append(i)
        

# Reverse sort list so that deleted indexes dont impact remaining indexes (delete high numbers first)
mev75_index_to_delete.sort(reverse=True)    
    
#%% 1.3: Take filtered ids (over 75 meV hull energy or deleted entry) and delete from Simmate table
    

mev75_mp_table = copy.copy(data_mp_table)
mev75_mp_list = copy.copy(data_mp_list) 


for i,x in enumerate(mev75_index_to_delete):
    mev75_mp_table = mev75_mp_table.drop(x)
    del mev75_mp_list[x]
    print("Done with number", i, "index",x)
    
# Reset index     
mev75_mp_table = mev75_mp_table.reset_index(drop=True)
mev75_mp_list = mev75_mp_list.reset_index(drop=True)

#%% 1.4: Confirm that the hull energy list index aligns with the Simmate table structure index 

hull_confirm = 0

for l,k in enumerate(mev75_mp_list):
    mpid2 = mev75_mp_table.id[i]
    corr_e_hull_test = mpr.thermo.get_data_by_id(mpid2).energy_above_hull
    if corr_e_hull_test != mev75_hull_energy[i]:
        print("Try again cowboy")
        hull_confirm = hull_confirm + 1
    else:
        print("Done with",l)
        
   
#%% 1.5: FILTER DOWN TO ONE STRUCTURE PER COMPOSITION BASED ON HULL ENERGY

#Filter repeated structures in original F list
   
structure_index_to_delete = []
structure_rf_to_delete = []
data_hull_energy = copy.copy(mev75_hull_energy)
data_reduced_form = []

# Make list of reduced formulas
for i,x in enumerate(mev75_mp_list):
    data_reduced_form.append(mev75_mp_table.formula_reduced[i])


for i,x in enumerate(data_reduced_form):
    count = data_reduced_form.count(x)
    if count > 1:
        indices_of_repeats = [i for i, m in enumerate(data_reduced_form) if m == x] 
        multi_hull_e = []
        for j,y in enumerate(indices_of_repeats):
          multi_hull_e.append(data_hull_energy[y])  
                
        for j,y in enumerate(indices_of_repeats):  
            if data_hull_energy[y] != min(multi_hull_e) and y not in structure_index_to_delete:   
                structure_index_to_delete.append(y)
                structure_rf_to_delete.append(data_reduced_form[y])        
        
        
 
# Reverse sort list so that deleted indexes dont impact remaining indexes
structure_index_to_delete.sort(reverse=True)        
      
# Vertify that each index to delete actually has a second structure  
check_ur_work = []        
for l,k in enumerate(structure_index_to_delete):
    count2 = data_reduced_form.count(data_reduced_form[k])
    if count2 > 1:
        check_ur_work.append("Checks out")
    else:
        check_ur_work.append("You messed up somewhere")
        
print("You messed up", check_ur_work.count("You messed up somwehere"),"times!")     
   
# Delete repeated structures from F list
data_mp_table2 = copy.copy(mev75_mp_table)
data_mp_list2 = copy.copy(mev75_mp_list) 

for h,f in enumerate(structure_index_to_delete):
    data_mp_table2 = data_mp_table2.drop(f)
    del data_mp_list2[f]


# Tables/lists to be taken forward and filtered for complete/partial deF
data_mp_table_complete = copy.copy(data_mp_table2)
data_mp_list_complete = copy.copy(data_mp_list2)
data_mp_table_partial = copy.copy(data_mp_table2)
data_mp_list_partial = copy.copy(data_mp_list2)

data_mp_table_complete = data_mp_table_complete.reset_index(drop=True)
data_mp_table_partial = data_mp_table_partial.reset_index(drop=True)


#%% 2.1: COMPLETE DEF PAIRS FILTER

print("Len of data_mp_list_complete:", len(data_mp_list_complete))

mp_index = 0

for struc in data_mp_list_complete:
    
    # Get id, energy per atom, spacegroup, and hull energy for F structure
    mpid_f = data_mp_table_complete.id[mp_index]
    energy_per_atom_f = data_mp_table_complete.energy_per_atom[mp_index]
    spacegroup_f = data_mp_table_complete.spacegroup[mp_index]
    e_hull_f = data_mp_table_complete.energy_above_hull[mp_index]
        
    # Remove F from each entry in data_mp_list
    # For each entry check if there is another MP entry with reduced formula (formula_reduced)
    # identical to the reduced fromula after removing F (struc.composition.reduced_formula)
    struc_deF = copy.copy(struc)
    struc_deF.remove_species("F")  
    s = MatProjStructure.objects.filter(
        formula_reduced=struc_deF.composition.reduced_formula,  
        energy_above_hull__lte=0.2,
    ).to_dataframe()  
    
    s_list = MatProjStructure.objects.filter(
        formula_reduced=struc_deF.composition.reduced_formula,  
        energy_above_hull__lte=0.2,
    ).to_toolkit()  
    
    
    # For deF matches identified in "s", get the id, energy per atom, spacegroup, and hull energy for deF structure
    # Then add the F/deF properties to the property lists
    if len(s_list) > 0:
        #i = 0
        print(len(s), "match for index", mp_index,":", struc_deF.composition.reduced_formula)
        
        s_multi_hull_e = []
        s_list_skip = []
        
        # Make list of CORRECT hull energies to compare for filtering repeat structure 
        for i,x in enumerate(s_list):
            try: 
                w = mpr.get_entry_by_material_id(s.id[i])
            except:
                s_list_skip.append(i)   # If structure ID isnt on MP, note which structure it is to delete
                print("No material (list propogation)")
                continue      
            s_correct_e_hull = mpr.thermo.get_data_by_id(s.id[i]).energy_above_hull
            s_multi_hull_e.append(s_correct_e_hull)
       
        # Delete the structures whose ID's aren't on MP
        for i,x in enumerate(s_list):
            if i in s_list_skip:
                s = s.drop(i)
                del s_list[i]
                
        s = s.reset_index(drop=True)    
        
        # Check if correct hull energy is lowest in list of hull energies and if it's less than 75mev
        for i,x in enumerate(s_list):
                
            print("\n2nd s_list length:",len(s_list),":hull list length",len(s_multi_hull_e))
            
            #Filter out repeat structures with hull energy
            if s_multi_hull_e[i] == min(s_multi_hull_e) and s_multi_hull_e[i] <= 0.075:
            
                mpid_def = s.id[i]
                energy_per_atom_def = s.energy_per_atom[i]
                spacegroup_def = s.spacegroup[i]
                e_hull_def = s.energy_above_hull[i]
          
                mpids_list_f.append(mpid_f)
                mpids_list_def.append(mpid_def)
                energy_per_atom_list_f.append(energy_per_atom_f)
                energy_per_atom_list_def.append(energy_per_atom_def)
                spacegroup_list_f.append(spacegroup_f)
                spacegroup_list_def.append(spacegroup_def)
                e_hull_list_f.append(e_hull_f)
                e_hull_list_def.append(e_hull_def)
                def_type.append("Complete")
          
    else:
        print("Zero matching for index:", mp_index)
    mp_index = mp_index + 1
    
#%% 2.2: Copy lists so that complete filter process doesnt have to run again

mpids_list_f_copy = copy.copy(mpids_list_f)
mpids_list_def_copy = copy.copy(mpids_list_def) 

energy_per_atom_list_f_copy = copy.copy(energy_per_atom_list_f) 
energy_per_atom_list_def_copy = copy.copy(energy_per_atom_list_def) 
spacegroup_list_f_copy =copy.copy(spacegroup_list_f) 
spacegroup_list_def_copy =copy.copy(spacegroup_list_def) 
e_hull_list_f_copy =copy.copy(e_hull_list_f) 
e_hull_list_def_copy = copy.copy(e_hull_list_def) 
def_type_copy = copy.copy(def_type)    

#%% 2.3: Restore from copied lists if problem occured

# mpids_list_f = copy.copy(mpids_list_f_copy)
# mpids_list_def = copy.copy(mpids_list_def_copy) 

# energy_per_atom_list_f = copy.copy(energy_per_atom_list_f_copy) 
# energy_per_atom_list_def = copy.copy(energy_per_atom_list_def_copy) 
# spacegroup_list_f =copy.copy(spacegroup_list_f_copy) 
# spacegroup_list_def =copy.copy(spacegroup_list_def_copy) 
# e_hull_list_f =copy.copy(e_hull_list_f_copy) 
# e_hull_list_def = copy.copy(e_hull_list_def_copy) 
# def_type = copy.copy(def_type_copy)   


#%% 3.1: PARTIAL DEF PAIRS: ENSURE REFERENCE TABLES ARE UPDATED
    
data_mp_table_partial = copy.copy(data_mp_table2)
data_mp_list_partial = copy.copy(data_mp_list2)
data_mp_table_partial = data_mp_table_partial.reset_index(drop=True)

#%%  3.2 PARTIAL DEF PAIRS FILTER 

print("Len of data_mp_list_partial:", len(data_mp_list_partial))

mp_index = 0

for struc in data_mp_list_partial:
    #print(struc.composition.reduced_formula)
    mpid2_f = data_mp_table_partial.id[mp_index]
    energy_per_atom2_f = data_mp_table_partial.energy_per_atom[mp_index]
    spacegroup2_f = data_mp_table_partial.spacegroup[mp_index]
    e_hull2_f = data_mp_table_partial.energy_above_hull[mp_index]
    
    # p1 filtering only catches structures that have the same chemical system, but not same ratio of atoms
    p1_list = MatProjStructure.objects.filter(  # NOTE: THIS EXCLUDES ALL FULLY DEFLUORINATED STRUCTURES
        elements=data_mp_table_partial.elements[mp_index],  # Check that partial deF entries have same chemical system as original
        # is_stable=True,
        energy_above_hull__lte=0.2,
        nelements__gt="2",
    ).exclude(nsites=struc.composition.num_atoms).to_toolkit()
    
    p1_table = MatProjStructure.objects.filter(  # NOTE: THIS EXCLUDES ALL FULLY DEFLUORINATED STRUCTURES
        elements=data_mp_table_partial.elements[mp_index],  # Check that partial deF entries have same chemical system as original
        # is_stable=True,
        energy_above_hull__lte=0.2,
        nelements__gt="2",
    ).exclude(nsites=struc.composition.num_atoms).to_dataframe()
    
    
    if len(p1_list) > 0:     
        #print(struc.composition.reduced_formula)
        elements1 = struc.composition.elements   
        struc_def = copy.copy(struc)
        struc_def.remove_species("F")
       # print(struc.composition.reduced_formula)
        
        p1_multi_hull_e = []
        p1_list_skip = []
        
        #i = 0
        print(len(p1_list),"matches for index:", mp_index, struc.composition.reduced_formula)
        
        # START 8.10.22 FILTER REPEAT STRUCTURE CODE
        for i,x in enumerate(p1_list):
            try: 
                w = mpr.get_entry_by_material_id(p1_table.id[i])
            except:
                p1_list_skip.append(i)
                continue
            p1_correct_e_hull = mpr.thermo.get_data_by_id(p1_table.id[i]).energy_above_hull
            p1_multi_hull_e.append(p1_correct_e_hull)
            
      
        for i,x in enumerate(p1_list):
            if i in p1_list_skip:
                p1_table = p1_table.drop(i)
                del p1_list[i]
      
        p1_table = p1_table.reset_index(drop=True)
      
        for i,x in enumerate(p1_list):
                     
            x_def = copy.copy(x)
            x_def.remove_species("F")
            
            struc_def_element1 = struc_def.composition.get_atomic_fraction(elements1[0])
            struc_def_element2 = struc_def.composition.get_atomic_fraction(elements1[1])
            x_def_element1 = x_def.composition.get_atomic_fraction(elements1[0])
            x_def_element2 = x_def.composition.get_atomic_fraction(elements1[1])
            struc_elementF = struc.composition.get_atomic_fraction(elements1[-1])
            x_elementF = x.composition.get_atomic_fraction(elements1[-1])
            
            
            
            # Filter to only include structures with the same ratio of non-F elements
            # F had to be removed from a copy of the structure so that ratios could be accurately compared
            # Also check that F structure has higher F fraction than deF
            if struc_def_element1 == x_def_element1 and struc_def_element1 == x_def_element1 and struc_elementF > x_elementF and p1_multi_hull_e[i] == min(p1_multi_hull_e) and p1_multi_hull_e[i] <= 0.075:                   
                print("A REAL MATCH!", struc.composition.reduced_formula, "-->", x.composition.reduced_formula)               
                print(struc_def_element1, x_def_element1)
                mpid2_def = p1_table.id[i]
                energy_per_atom2_def = p1_table.energy_per_atom[i]
                spacegroup2_def = p1_table.spacegroup[i]
                e_hull2_def = p1_table.energy_above_hull[i]
              
                mpids_list_f.append(mpid2_f)
                mpids_list_def.append(mpid2_def)
                energy_per_atom_list_f.append(energy_per_atom2_f)
                energy_per_atom_list_def.append(energy_per_atom2_def)
                spacegroup_list_f.append(spacegroup2_f)
                spacegroup_list_def.append(spacegroup2_def)
                e_hull_list_f.append(e_hull2_f)
                e_hull_list_def.append(e_hull2_def)
                def_type.append("Partial")
            
    else:
        print("No match for index:", mp_index)
    mp_index = mp_index + 1 




#%% 3.3: Copy lists so that partial filter process doesnt have to run again

mpids_list_f_copy = copy.copy(mpids_list_f)
mpids_list_def_copy = copy.copy(mpids_list_def)   

energy_per_atom_list_f_copy = copy.copy(energy_per_atom_list_f) 
energy_per_atom_list_def_copy = copy.copy(energy_per_atom_list_def) 
spacegroup_list_f_copy =copy.copy(spacegroup_list_f) 
spacegroup_list_def_copy =copy.copy(spacegroup_list_def) 
e_hull_list_f_copy =copy.copy(e_hull_list_f) 
e_hull_list_def_copy = copy.copy(e_hull_list_def) 
def_type_copy = copy.copy(def_type)    
    
    
#%% 4.1: Instantiate each ID as a Material object

materials_list_f = []
materials_list_def = []

# Create list of Material objects for F and DeF structures
for i,x in enumerate(mpids_list_f):
    y = mpids_list_def[i]
    materials_list_f.append(Material(x))
    materials_list_def.append(Material(y))
    print("\n Material creation: Done with index",i,x)

#%% 4.2: CALCULATE PROPERTIES FOR EACH MATERIAL OBJECT

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
vol_e_density_list = []
grav_e_density_list = []
atoms_per_fu_f =[]
atoms_per_fu_def =[]
corrected_e_hull_list_f = []
corrected_e_hull_list_def = []
bandgap_list_f = []
bandgap_list_def = []


for i,x in enumerate(materials_list_f):
    y = materials_list_def[i]
    j = mpids_list_f[i]
    k = mpids_list_def[i]
    reduced_formula_f.append(x.reduced_formula)
    reduced_formula_def.append(y.reduced_formula)
    composition_list_f.append(x.structure.composition)
    composition_list_def.append(y.structure.composition)
    f_transfer_list.append(FTransfer(x,y))
    voltage_list.append(Voltage(x, y, energy_per_atom_list_f[i], energy_per_atom_list_def[i], f_transfer_list[i]))
    volume_per_f_list.append(VolumePerF(x, f_transfer_list[i]))
    volume_change_list.append(VolumeChange(x,y))
    density_list_f.append(x.density)
    atoms_per_fu_f.append(x.atoms_per_fu)
    atoms_per_fu_def.append(y.atoms_per_fu)
    corrected_e_hull_list_f.append(x.corrected_hull_energy)
    corrected_e_hull_list_def.append(y.corrected_hull_energy)
    bandgap_list_f.append(BandGap(j))
    bandgap_list_def.append(BandGap(k))
    
    
    
    e_per_A3 = 1/volume_per_f_list[i]
    C_per_A3 = e_per_A3 * 1.60217e-19
    J_per_A3 = C_per_A3 * voltage_list[i]
    J_per_L = J_per_A3 * 1e27
    Wh_per_L = J_per_L / 3600
    vol_e_density_list.append(Wh_per_L)
    
    grav_e_density = vol_e_density_list[i] / density_list_f[i]
    grav_e_density_list.append(grav_e_density)
    
    print("Other properties complete for index:",i)
    


#%% 4.3: INTERFACE REACTIONS

from pymatgen.ext.matproj import MPRester  # This version needed for interface rxns
mpr = MPRester("KCWNf75lwOINtIUANaNCP9Xfdof52mTC")


products_list= []
direct_rxn = []

for i,x in enumerate(reduced_formula_f): 
    
        print("Index:",i, reduced_formula_f[i], "-->", reduced_formula_def[i])
        prod = mpr.get_interface_reactions(                     # Perform interface rxn with F and de-F structure
            reduced_formula_f[i], reduced_formula_def[i],  relative_mu=-1, use_hull_energy=False
        )
        print(prod[1])
        reaction = prod[1].get('rxn')              # Get the 'rxn' string (reactions with different 'rxn' and 'rxn_str' strings do not have a 'products' entry)
        reaction_text = f'{reaction}'          # Turn rxn string into text so that it can be partitioned into products string
        reaction_products = reaction_text.partition("> ")[2]        # Get text string of products after the reaction arrow 
        print("Reaction products", reaction_products, "\n")
        products_list.append(reaction_products)      # Add reaction_products to "master" list of products
                                             
        for entry in products_list:                  # If products equals either reactant (F or de-F), then the reaction is direct (Yes), otherwise, a reaction occured so it is not direct (No)
           if products_list[i] == reduced_formula_f[i] or products_list[i] == reduced_formula_def[i]:
               direct_rxn.append('Yes')
               break
           else:
               direct_rxn.append('No')
               break


#%% 4.4: COST ANALYSIS
    
cost_database = CostDBCSV('costdb_elements.csv')
cost_analyzer = CostAnalyzer(cost_database)

composition = Composition("Ca2NCl")

cost_per_kg_list_f = []
cost_per_mol_list_f = []
cost_per_kg_list_def = []
cost_per_mol_list_def = []

for i,x in enumerate(composition_list_f):
    y = composition_list_def[i]
    
    # Get cost/kg and cost/mol for each F and deF
    cost_per_kg_f = cost_analyzer.get_cost_per_kg(x) # try x.reduced_composition
    cost_per_mol_f = cost_analyzer.get_cost_per_mol(x) / atoms_per_fu_f[i]
    cost_per_kg_def = cost_analyzer.get_cost_per_kg(y)
    cost_per_mol_def = cost_analyzer.get_cost_per_mol(y) / atoms_per_fu_def[i]
    
    # Add costs to appropriate list
    cost_per_kg_list_f.append(cost_per_kg_f)
    cost_per_mol_list_f.append(cost_per_mol_f)
    cost_per_kg_list_def.append(cost_per_kg_def)
    cost_per_mol_list_def.append(cost_per_mol_def)
    print(i)

#%%
import math

li_comp = Composition("LiCoO2")

li_cost_p_kg = cost_analyzer.get_cost_per_kg(li_comp)
li_cost_p_mol = cost_analyzer.get_cost_per_kg(li_comp) / 4

log_li_cost_p_mol= math.log(li_cost_p_mol)

#%% 4.5: ACTIVATION ENERGY

activation_energy_table = pd.read_csv('F_transport_activation_energies.csv')

activation_ids = activation_energy_table['structure_id'].tolist()
activation_energies = activation_energy_table['approx_barrier_corrected'].tolist()

activation_energy_list_f = []
activation_energy_list_def = []

# Count number of F structures with activation energy (303)
for i,x in enumerate(mpids_list_f):
    if x in activation_ids:
        activation_energy_list_f.append(activation_energies[i])
        print("Match for", x)
    else:
        activation_energy_list_f.append("-")
        print("no match")
        
# Count number of def structres with activation energy (O)        
for i,x in enumerate(mpids_list_def):
    if x in activation_ids:
        activation_energy_list_def.append(activation_energies[i])
        print("Match for", x)
    else:
        activation_energy_list_def.append("-")
        print("no match")


#%% 5.1: CREATE TABLE WITH ALL INFORMATION/PROPERTIES

complete_def_pairs_table = pd.DataFrame({
    'F form': reduced_formula_f,
    'Def form': reduced_formula_def,
    'Voltage': voltage_list,
    'Volume Per F(A^3)': volume_per_f_list,
    'Percent Expansion': volume_change_list,
    'Vol Energy Density (Wh/L)': vol_e_density_list,
    'Grav Energy Density (Wh/kg)': grav_e_density_list,
    'F transfer': f_transfer_list,
    'F cost/kg': cost_per_kg_list_f,
    'DeF cost/kg': cost_per_kg_list_def,
    'F cost/mol': cost_per_mol_list_f,
    'DeF cost/mol': cost_per_mol_list_def,
    'F ID': mpids_list_f,
    'DeF ID': mpids_list_def,
    'F spacegroup': spacegroup_list_f,
    'DeF spacegroup': spacegroup_list_def,
    'F E_hull orig': e_hull_list_f,
    'F E_hull corr': corrected_e_hull_list_f,
    'DeF E_hull orig': e_hull_list_def,
    'DeF E_hull corr': corrected_e_hull_list_def,
    'F bandgap': bandgap_list_f,
    'DeF bandgap': bandgap_list_def,
    'F transport barrier': activation_energy_list_f,
    'DeF transport barrier': activation_energy_list_def,
    'Direct rxn': direct_rxn,
    'Products': products_list,
    'DeF type': def_type,
    })

complete_def_pairs_table.to_csv('complete_def_cathode_list_v9.csv')



#%% Testing


list2 = MatProjStructure.objects.filter(
    elements__icontains='"F"',
    # is_stable=True,
    energy_above_hull__lte=0.2,
    nelements__gt="3",
).to_dataframe()

#hull = mpr.thermo.get_data_by_id(g).energy_above_hull   
