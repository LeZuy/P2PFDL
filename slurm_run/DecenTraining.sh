#!/bin/bash

# Sample slurm submission script for the Chimera compute cluster
# Lines beginning with # are comments, and will be ignored by
# the interpreter.  Lines beginning with #SBATCH are directives
# to the scheduler.  These in turn can be commented out by
# adding a second # (e.g. ##SBATCH lines will not be processed
# by the scheduler).
#
#
# set name of job
#SBATCH --job-name=P2PFDL
#
# set the number of processors/tasks needed
# for hyperthreaded,shared memory jobs, set 1 task, 1 node, and 
# set --cpus-per-task to total number of threads, otherwise
# set -n to number of processors/tasks
##SBATCH -n 4
#SBATCH -n 1
#SBATCH --cpus-per-task=4

# set the number of Nodes needed.  Set to 1 for shared memory jobs
#SBATCH -N 1

# set an account to use
# if not set, per-user default will be used
# for scavenger users, use this format:
##SBATCH --account=pi_first.last
# for contributing users, use this format:
##SBATCH --account=[deptname|lastname|accountname]

# set max wallclock time  DD-HH:MM:SS
#SBATCH --time=00-12:00:00

# set a memory request
#SBATCH --mem=16gb

# Set filenames for stdout and stderr.  %j can be used for the jobid.
# see "filename patterns" section of the sbatch man page for
# additional options
#SBATCH --error=./results/data.err
#SBATCH --output=./results/data.out
#

# set the partition where the job will run.  Multiple partitions can
# be specified as a comma separated list
# Use command "sinfo" to get the list of partitions
##SBATCH --partition=Intel6240
#SBATCH --partition=Intel6240,Intel6248,Intel6326,Intel6126,Intel2650
# restricting inheritance of environment variables is required for chimera12 and 13:
# if this option is used, source /etc/profile below.
#SBATCH --export=HOME

#Optional
# mail alert at start, end and/or failure of execution
# see the sbatch man page for other options
##SBATCH --mail-type=ALL
# send mail to this address
##SBATCH --mail-user=first.last@umb.edu

# Put your job commands here, including loading any needed
# modules or diagnostic echos.

# this job simply reports the hostname and sleeps for two minutes

# source the local profile.  This is recommended in conjunction
# with the --export=HOME or --export=NONE sbatch options.
. /etc/profile

echo "Job ID:               $SLURM_JOB_ID"
echo "Job Name:             $SLURM_JOB_NAME"
echo "User:                 $SLURM_JOB_USER"
echo "Partition:            $SLURM_JOB_PARTITION"
echo "Node List:            $SLURM_JOB_NODELIST"
echo "Number of Nodes:      $SLURM_JOB_NUM_NODES"
echo "Tasks (ntasks):       $SLURM_NTASKS"
echo "CPUs per Task:        $SLURM_CPUS_PER_TASK"
echo "CPUs on Node:         $SLURM_CPUS_ON_NODE"
echo "CPUs allocated:       $SLURM_JOB_CPUS_PER_NODE"
echo "Memory allocated:     $SLURM_MEM_PER_NODE"
echo "Submit Directory:     $SLURM_SUBMIT_DIR"
echo "Hostname:             $(hostname)"
echo "Time:           $(date)"


hostname
# Diagnostic/Logging Information
# python main.py --consensus mean
# python main.py --consensus tverberg
# python main.py --consensus geomedian
# python ./model/train.py
python ./scripts/example_usage.py
echo "Finish Run"
echo "end time is `date`"