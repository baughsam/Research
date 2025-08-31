from ase.io import read, write
from ase.build import surface

bulk = read('Anatase.cif')

output_dir = './'

for i in range(1, 9):
    slab = surface(bulk, (1, 1, 1), layers=i, vacuum=20.0)

    slab.center(axis=2)

    filename = f'{output_dir}/Anatase_111_{i}.xyz'
    write(filename, slab)
    print(f'Wrote: {filename}')
