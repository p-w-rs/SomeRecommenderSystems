#!/bin/bash

#SBATCH --job-name="push"
#SBATCH --output=job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=powersj@msoe.edu
#SBATCH --partition=teaching
#SBATCH --nodes=1

## SCRIPT START

git add .
git commit -m "stuff"
git push -u origin main

## SCRIPT END
