sudo apt-get update
sudo apt-get install swig

# linux dependencies
sudo apt-get install -y libboost-regex-dev libboost-system-dev libboost-thread-dev libboost-program-options-dev
sudo apt-get install -y zlib1g-dev

mkdir build
cd build
cmake -DPYTHON3_API=ON ..
sudo make install

export PYTHONPATH="${PYTHONPATH}:/usr/local/share/freeling/APIs/python3"

