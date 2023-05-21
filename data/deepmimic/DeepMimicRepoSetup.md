# Steps to Set up the [DeepMimic 2018 Repo](https://github.com/xbpeng/DeepMimic)

To set up the DeepMimic 2018 Repo in Linux based systems:
1. Clone repository to ```REPO_PATH```
```
git clone git@github.com:xbpeng/DeepMimic.git
```

2. Install following Python dependencies:
```
pip install PyOpenGL PyOpenGL_accelerate
pip install tensorflow
pip install mpi4py
```
**Note**: Installation of these dependencies directly in your system without setting up a conda or virtual environment is 
recommended as the virtual environment later on cause lot of issues with path dependencies. 

3. Build the dependencies 

In ```DeepMimicCore/Makefile``` modify the Python path dependencies according to your python version
```
PYTHON_INC = /usr/include/python<version> #/usr/include/python3.10
PYTHON_LIB = /usr/lib/ -lpython<version> #/usr/lib/ -lpython3.10
```
If the path dependency is not correct then ```Python.h not found``` error occurs. Next,

```
cd REPO_PATH
./DeepMimicCore/build.sh
```
This should generate a ```DeepMimicCore/DeepMimic.py``` file. \
Possible errors that can be encountered during the process and their solutions: ``freeglut: build fails with multiple definitions```: This happens due to gcc version issues. Modify the code in required files according to this 
[solution](https://gitweb.gentoo.org/repo/gentoo.git/tree/media-libs/freeglut/files/freeglut-3.2.1-gcc10-fno-common.patch?id=f9102571b69d9fc05471a592fda252681fdfdef1)

4. Now try running the command: ```python3 DeepMimic.py --arg_file args/play_motion_humanoid3d_args.txt``` \
Possible errors and solutions: \
- ```ImportError: libGLEW.so.2.1: cannot open shared object file: No such file or directory search for libGLEW.so.2.1```
Solution:
```
sudo ln -s /path/to/libGLEW.so.2.1  /usr/lib/x86****/libGLEW.so.2.1 
sudo ln -s /path/to/libGLEW.so.2.1.0  /usr/lib/x86****/libGLEW.so.2.1.0
```
- ```ImportError: libBulletDynamics.so.2.88: cannot open shared object file: No such file or directory```
Solution: ```export LD_LIBRARY_PATH=/usr/local/lib/``` ( can be temporary when run in terminal) 
(libBullet file are present in that path - gets installed in that path after the command sudo make install while installing Bullet)
- There might be a Tensorflow version mismatch problem. Since the codebase uses tensorflow v1.12.0 you can use [Tensorflow Migrate](https://www.tensorflow.org/guide/migrate/upgrade)
to update and your codebase to Tensorflow v2.

5. To load and play a mocap clip: ```python DeepMimic.py --arg_file args/play_motion_humanoid3d_args.txt``` \
Change <action> in the line ```--motion_file data/motions/humanoid3d_<action>.txt``` inside the ```play_motion_humanoid3d_args.txt``` to play the desired clip.





