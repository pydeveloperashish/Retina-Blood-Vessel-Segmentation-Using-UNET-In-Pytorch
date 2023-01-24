echo [$(date)]: "START"
echo [$(date)]: "Creating conda env with python 3.8"
conda create --prefix ./venv python==3.8 -y
echo [$(date)]: "Activating Conda env"
source activate ./venv
conda activate ./venv
echo [$(date)]: "Installing pip requirements..."
pip3 install -r requirements.txt
echo [$(date)]: "END"



# Install this project using command:- bash init_setup.sh