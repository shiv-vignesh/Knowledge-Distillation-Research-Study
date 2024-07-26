#!/bin/bash -l
######## Declaring Slurm Configuration Options

### Accounting Conf: Remain same for all jobs
#SBATCH --job-name=debugg_full_run_v3
#SBATCH --comment="Partial test; bs 8, trainer grad accumilation 12: reduced the num of grad acc bs from 12 to 6. Debugged for errors in v2"
#SBATCH --account=kdpt-llm
#SBATCH --partition=debug
#SBATCH --time=1-00:00:00

### Job Output Conf
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

##Slack Conf
#SBATCH --mail-user=slack:@sk4858
#SBATCH --mail-type=ALL

##Node Conf
#SBATCH --nodes=1

##GPU & CPU Conf
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1


##Memory Conf
#SBATCH --mem=10g

# Loading Software/Libraries
spack env activate default-ml-24050101

# Running Code
python -u /home/sk4858/KD_research_project/minillm.py