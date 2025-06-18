# ParaCuda

## Overview

**ParaCuda** is a **para**llel **cuda** execution framework for arbitrary tools. Consider it a (very) poor man's version of workflow
management systems. ParaCuda uses a config file that describes the script that needs to be run and the parameters
that we want to pass down to the script. ParaCuda then logs the execution and distributes it over all specified GPUs.

## Motivation

In machine learning, or modern bioinformatics we often require GPUs for execution of processes or predictions. Sometimes,
this is extremely trivial and a simple solution is desired that can run a script parameterized in different ways across
all available GPUs. ParaCuda addresses this challenge by easy set-up and execution of experiments concurrently.

## Installation

For the moment, clone the repository and create the environment using mamba.

```bash
git clone https://github.com/gieses/paracuda.git
mamba env create -f environment.yml
```
