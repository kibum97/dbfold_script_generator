import os
import glob
import natsort
from dbfold.utils import *
import yaml
from math import ceil
from jinja2 import Template
import mdtraj as md
import numpy as np
import subprocess
import re
import shutil
import warnings

script_path = os.path.dirname(os.path.abspath(__file__))

class Simulation:
    def __init__(self, pdbroot, identifier, sim_path, mcpu_path, umbrella=False, temperature_replica=False, constraint=False):
        self.pdbroot = pdbroot
        self.identifier = identifier
        self.mcpu_path = mcpu_path
        self.sim_path = sim_path
        os.makedirs(os.path.join(self.sim_path, self.identifier, 'MCPU_run'), exist_ok=True)
        self.umbrella = umbrella
        self.temperature_replica = temperature_replica
        self.constraint = constraint
        if umbrella:
            print('Umbrella sampling is enabled.')
        if temperature_replica:
            print('Temperature replica exchange is enabled.')
        if constraint:
            print('Constraints are enabled.')
        # Set mcpu related paths
        self.protein_parameter_path = os.path.join(self.sim_path, self.pdbroot)
        self.mcpu_config_path = os.path.join(self.mcpu_path, 'config_files')
        self.mcpu_exec_path = os.path.join(self.mcpu_path,'src_mpi_umbrella/fold_potential_mpi')

    def count_contacts_rounded(self, cutoff):
        # Count number of native contacts
        if not os.path.isfile(self.native_pdb):
            print("count_contacts: No file at {}".format(self.native_pdb))
            print("Assuming possible dry run")
            return 0
        protein = md.load(self.native_pdb)
        dist, pair = md.compute_contacts(protein, scheme='ca')
        new_dist = []
        new_pair = []
        for i, p in enumerate(pair):
            if abs(p[0] - p[1]) < 8:
                new_pair.append(p)
                new_dist.append(dist[0, i] * 10)
        contact_count = np.sum(np.array(new_dist) < cutoff)
        contact_count = round(np.array(contact_count) / 10) * 10
        self.native_contact_count = contact_count
        self.contact_distance_cutoff = cutoff
        return contact_count
    
    def compute_protein_parameters(self, recompute=False):
        # Check if the protein parameter file already exists
        if os.path.exists(os.path.join(self.sim_path,f'{self.pdbroot}.sec_str')):
            print(f'Protein parameter files already exist. Skipping computation. If you want to recompute, please set recompute=True')
            return 0
        else:
            native = md.load(self.native_pdb)
            n_residues = native.n_residues
            # FASTA
            self.fasta = native.topology.to_fasta()[0]
            with open(os.path.join(self.sim_path,f'{self.pdbroot}.fasta'), 'w') as f:
                f.write(f'>{self.pdbroot}\n')
                for i in range(0, len(self.fasta), 80):
                    f.write(self.fasta[i:i+80] + '\n')
            f.close()
            
            # Triple and Sidechain Torsion
            os.symlink(f"{self.mcpu_path}/mcpu_prep/sct.energy", f"{self.sim_path}/sct.energy")
            os.symlink(f"{self.mcpu_path}/mcpu_prep/triple.energy", f"{self.sim_path}/triple.energy")
            save_triple_path = os.path.join(self.mcpu_path, "mcpu_prep", "save_triple")
            subprocess.run([save_triple_path, self.pdbroot], cwd=self.sim_path, check=True)
            os.unlink(f"{self.sim_path}/sct.energy")
            os.unlink(f"{self.sim_path}/triple.energy")

            # Secondary Structure
            with open(os.path.join(self.sim_path,f'{self.pdbroot}.sec_str'),'w') as f:
                f.write(f'{"0" * n_residues}\n')
                f.write(f'{"C" * n_residues}\n')
            f.close()

    def set_starting_structure(self, starting_pdb):
        self.starting_pdb = starting_pdb
    
    def set_native_structure(self, native_pdb):
        self.native_pdb = native_pdb

    def set_umbrella_sampling(self, umbrella_max, replica_steps, umbrella_bias=0.02, umbrella_spacing=10, max_exchange=75):
        if self.umbrella:
            self.umbrella_bias = umbrella_bias
            self.umbrella_max = umbrella_max
            self.umbrella_spacing = umbrella_spacing
        else:
            raise ValueError('Umbrella sampling is not enabled. Please set umbrella=True when initializing the Simulation class.')
        if not self.temperature_replica:
            self.replica_steps = replica_steps
            self.max_exchange = max_exchange
        else:
            warnings.warn('Temperature replica exchange is enabled. The number of replica steps and max exchange pairs will be set in the set_temperature_replica method.')
        
    def set_temperature(self, tempearture):
        if self.temperature_replica:
            raise ValueError('Temperature replica exchange is enabled. Please set the temperature in the set_temperature_replica method.')   
        else:
            self.min_temperature = tempearture
            self.temperature_spacing = 0.025

    def set_temperature_replica(self, min_temperature, temperature_spacing, num_temperature, replica_steps, max_exchange=75):
        if self.temperature_replica:
            self.min_temperature = min_temperature
            self.temperature_spacing = temperature_spacing
            self.num_temperature = num_temperature
            self.replica_steps = replica_steps
            self.max_exchange = max_exchange
            print(f'Maximum tempareture will be {min_temperature + (num_temperature - 1) * temperature_spacing}')
        else:
            raise ValueError('Temperature replica exchange is not enabled. Please set temperature_replica=True when initializing the Simulation class.')
  
    def set_montecarlo_parameters(self, mc_steps, log_steps):
        self.mc_steps = mc_steps
        self.log_steps = log_steps  

    def set_constraint(self, k_constraint, constraint_file):
        self.k_constraint = k_constraint
        self.constraint_file = constraint_file

    def set_slurm_resources(
            self,
            job_name='dbfold',
            partition='sapphire',
            time='3-00:00',
            memory='900G',
            cpu_per_node=112,
            email=None,
        ):
        self.job_name = job_name
        self.partition = partition
        self.time = time
        self.memory = memory
        self.cpu_per_node = cpu_per_node
        self.email = email

    def generate_config(self, config_path, checkpoint=None):
        self.node_per_temperature = (self.umbrella_max//self.umbrella_spacing if self.umbrella else 1) + 1
        self.number_of_replicas = self.node_per_temperature * (self.num_temperature if self.temperature_replica else 1)
        if not hasattr(self, 'template_pdb'):
            with open(os.path.join(self.sim_path,f'nothing.template'),'w') as f:
                f.close()
            self.template_pdb = os.path.join(self.sim_path,f'nothing.template')
        with open(os.path.join(script_path, 'config.template'), 'r') as f:
            teamplate_str = f.read()
        template = Template(teamplate_str)
        config = {
            # protein configuration
            'starting_file': self.starting_pdb,
            'native_file': self.native_pdb,
            'template_file': self.template_pdb if hasattr(self, 'template_pdb') else None,
            'checkpoint_directory' : self.checkpoint_directory if hasattr(self, 'checkpoint_directory') else None,
            'output_path': os.path.join(self.sim_path, self.identifier, f'MCPU_run/{self.pdbroot}'),
            # simulation parameters
            'montecarlo_steps': self.mc_steps,
            'montecarlo_log_interval': self.log_steps,
            # temperature replica exchange parameters
            'min_temperature': self.min_temperature if hasattr(self, 'min_temperature') else None,
            'temperature_spacing': self.temperature_spacing if hasattr(self, 'temperature_spacing') else None,
            'replica_exchange_steps': self.replica_steps if hasattr(self, 'replica_steps') else None,
            'max_exchange_pairs': self.max_exchange if hasattr(self, 'max_exchange') else None,            
            # umbrella sampling parameters
            'umbrella': self.umbrella,
            'umbrella_bias': self.umbrella_bias if hasattr(self, 'umbrella_bias') else None,
            'umbrella_max': self.umbrella_max if hasattr(self, 'umbrella_max') else None,
            'umbrella_spacing': self.umbrella_spacing if hasattr(self, 'umbrella_spacing') else None,
            'contact_distance_cutoff': self.contact_distance_cutoff if hasattr(self, 'contact_distance_cutoff') else None,
            'nodes_per_temperature': self.node_per_temperature if hasattr(self, 'node_per_temperature') else None,
            # constraints parameters
            'constraint': self.constraint,
            'conatraint_file': self.constraint_file if hasattr(self, 'constraint_file') else None,
            'k_constraint': self.k_constraint if hasattr(self, 'k_constraint') else None,
            # MCPU configuration
            'protein_dependent_param': self.protein_parameter_path,
            'mcpu_config_path': os.path.join(self.mcpu_path, 'config_files'),            
            'use_cluster': self.use_cluster_move if hasattr(self, 'use_cluster_move') else None,
            'max_cluster_move_steps': self.max_cluster_move_steps if hasattr(self, 'max_cluster_move_steps') else None,
        }
        rendered = template.render(config)
        self.config_path = config_path
        with open(config_path, 'w') as f:
            f.write(rendered)
    
    def generate_submission_script(self, bash_path, executable_path=None):
        if executable_path is None:
            executable_path = os.path.join(self.mcpu_path, 'src_mpi_umbrella/fold_potential_mpi')
        else:
            if not os.path.isabs(executable_path):
                executable_path = os.path.join(self.mcpu_path, executable_path)
        with open(os.path.join(script_path, 'sbatch.template'), 'r') as f:
            template_str = f.read()
        template = Template(template_str)
        config = {
            'nodes': ceil(self.number_of_replicas / self.cpu_per_node),
            'number_of_replicas': self.number_of_replicas,            
            'partition': self.partition,
            'time': self.time,
            'memory': self.memory,
            'job_name': self.job_name,
            'email': self.email,
            'working_directory': os.path.join(self.mcpu_path, 'src_mpi_umbrella/'),
            'mcpu_exec_path': executable_path,
            'config_path': self.config_path,
        }
        rendered = template.render(config)
        with open(bash_path, 'w') as f:
            f.write(rendered)
        self.submission_script = bash_path
        return 0
    
    def create_checkpoint(self, traj_directory, checkpoint_directory):
        self.checkpoint_directory = checkpoint_directory
        # Check if the directory exists
        if not os.path.exists(traj_directory):
            raise ValueError(f'The directory {traj_directory} does not exist.')
        # Check if the checkpoint directory exists
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory, exist_ok=True)
        # Get maximum montecarlo steps performed
        max_mc_step = self.get_max_mc_step(traj_directory)
        print(f'Maximum montecarlo step is {max_mc_step}')
        # Process and create checkpoint files based on log files
        log_files = glob.glob(os.path.join(traj_directory, '*.log'))
        print(f'Found {len(log_files)} log files in {traj_directory}')
        for log_file in log_files:
            if self.umbrella:
                match = re.match(r'^\.?/?(.+)_T_([0-9\.]+)_([0-9]+)\.log$', os.path.basename(log_file))
                protein, temperature, replica = match.groups()
                checkpoint_file = os.path.join(traj_directory, f"{protein}_{temperature}_{replica}.{max_mc_step}")
            else:
                match = re.match(r'^\.?/?(.+)_T_([0-9\.]+)\.log$', os.path.basename(log_file))
                protein, temperature = match.groups()
                checkpoint_file = os.path.join(traj_directory, f"{protein}_{temperature}.{max_mc_step}")
            with open(log_file, 'r') as file:
                for line in file:
                    if 'myrank' in line:
                        parts = line.split(',')
                        myrank = parts[0].split(':')[-1].strip()
                        break
                else:
                    print(f"No 'myrank' found in {log_file}")
                    continue
            output_file = os.path.join(checkpoint_directory, f"{myrank}.pdb")

            if os.path.exists(checkpoint_file):
                shutil.copy(checkpoint_file, output_file)
            else:
                print(f"Input file does not exist: {checkpoint_file}")
        return 0

    def print_config(self):
        with open(self.config_path, 'r') as f:
            print(f.read())
        return 0

    def print_slurm(self):
        with open(self.submission_script, 'r') as f:
            print(f.read())
        return 0

    def sumbit_job(self, scheduler='slurm'):
        if scheduler == 'slurm':
            self.submit_slurm()

    def submit_slurm(self):
        return os.system(f'sbatch {self.submission_script}')
    
    def get_max_mc_step(self, directory):
        pattern = re.compile(r'^.*\.(\d+)$')  
        max_step = None
        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                mc_step = int(match.group(1))
                if max_step is None or mc_step > max_step:
                    max_step = mc_step
        return max_step
    