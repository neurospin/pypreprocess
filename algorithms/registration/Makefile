all: lib

cython:
	echo "Compiling .pyx -> .c..."
	cython -a spm_hist2py.pyx
	echo "...done."

lib: clean cython
	echo "Building *.o -> spm_hist2.so..."
	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
	-I/usr/include/python2.7 -o spm_hist2py.so spm_hist2.c spm_hist2py.c
	echo "...done."

clean:
	rm -rf *.o *.so
	 