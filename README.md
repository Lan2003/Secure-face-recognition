Privacy-preserving multi-pose face recognition system

A multi-pose face recognition method based on deep learning is used and applied to a face recognition system based on homomorphic encryption to realize secure multi-pose face recognition.
1. Installation of model training environment
The code of the model training part is mainly based on Python and PyTorch, which is used to obtain multi-angle and occluded face recognition, that is, multi-pose face recognition. Hardware requirements support NVIDIA graphics card of CUDA to speed up training.
(1) It mainly depends on the library, and conda installation is recommended.
python=3.6.7
pytorch=1.8.1
torchvision=0.9.1
cudatoolkit=10.2.89
lmdb=1.2.0
pyarrow=0.17.0
(2) Installation of additional dependencies
pip install numpy opencv-python pillow lmdb
(3) Data set used
The training is mainly based on two data sets, including a large-scale face data set CASIA-WebFace and an LFW data set including different postures, lighting, expressions and ages. And because there is no open source special occluded face data set, random occlusion is added to each data set to construct a special occluded face data set. In total, the following data sets were used:
CASIA-WebFace face dataset, Occ-WebFace dataset with random occlusion;
LFW data set, CPLFW data set (LFW data set with more obvious angle difference)
Occ-LFW dataset with random occlusion;
(4) Training model
Run the following command to start the training:
bash run.sh
2. Installation of system operating environment
This system uses Python as the main programming language, the back end is based on Flask framework, and the front end uses HTML, CSS and JavaScript languages, with MySQL database. Can be used in any operating system. The requirment.txt in the reference code installs related dependencies.
(1) Upgrade the software warehouse
sudo apt update
sudo apt upgrade
(2) Clone the project to the local area
git clone https://github.com/Lan2003/Secure-face-recognition
(3) Upgrade related dependent warehouses
sudo apt install --no-install-recommends git cmake ninja-build gperf \
ccache dfu-util device-tree-compiler wget \
python3-dev python3-pip python3-setuptools python3-tk python3-wheel xz-utils file \
make gcc gcc-multilib g++-multilib libsdl2-dev libmagic1
(4) Install the main dependency library
pip3 install Flask==1.0.2 Flask-SQLAlchemy==2.4.0 Flask-WTF==0.14.2 mysqlclient==1.4.2.post1 virtualenv>=16.5.0 numpy==1.16.2 tensorflow==1.7.0 scipy== 1.2.1 scikit-learn==0.21.1 opencv-python h5py matplotlib Pillow requests psutil
(5) Configure the database
The database version requirement here is Mysql 5.7.X Firstly, the database structure in documents/face.sql is imported into MySQL5.7 database, and then the database configuration in config.py is modified. This database is used to store the ciphertext of face features and acts as a face database stored by the server in face recognition.
(6) run pythonmain.py.
