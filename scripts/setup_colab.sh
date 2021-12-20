# Install conda.
MINICONDA_INSTALLER_SCRIPT=Miniconda3-latest-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local
if ! test -f "$MINICONDA_INSTALLER_SCRIPT"; then
    wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
fi
chmod +x $MINICONDA_INSTALLER_SCRIPT
./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX

# Update conda
conda install --channel defaults conda python=3.9 --yes
conda update --channel defaults --all --yes

# Install dependencides
conda env create -f environment-gpu.yml
source activate condaenv &&
    pip install -e .
