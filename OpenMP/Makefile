#
# Computaci�n paralela (Grado Ingenier�a Inform�tica)
#
# Makefile: Contar cuerpos celestes
#

CC=gcc
CFLAGS=-O3 -fopenmp

EXES=ScanSky_OpenMP ScanSky_OpenMP_debug ScanSky_OpenMP_write

all: $(EXES)

clean:
	rm -f $(EXES)

ScanSky_OpenMP: ScanSky_OpenMP.c cputils.h
	$(CC) $(CFLAGS) $< -o $@

ScanSky_OpenMP_debug: ScanSky_OpenMP.c cputils.h
	$(CC) $(CFLAGS) -DDEBUG -DWRITE $< -o $@

ScanSky_OpenMP_write: ScanSky_OpenMP.c cputils.h
	$(CC) $(CFLAGS) -DWRITE $< -o  $@
