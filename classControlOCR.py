
import numpy as np
import string

class claseControladorReconocimientoDeMatriculas:

    def __init__(self, threshold, batch_size, dim_fil, dim_col, learning_ratio, rpi, h5):
        
        self.threshold = threshold
        self.batch_size = batch_size

        self.dim_fil = dim_fil
        self.dim_col = dim_col

        self.learning_ratio = learning_ratio

        self.rpi = rpi

        self.h5 = h5

        _dict = [v for v in string.ascii_uppercase]
        for i in range(10):
            _dict.append(str(i))
        self.dict = _dict

clasMatOcr = claseControladorReconocimientoDeMatriculas(threshold = 0.5,
                                                        batch_size = 5,
                                                        dim_fil = 32, dim_col = 128,
                                                        learning_ratio = 1e-3,
                                                        rpi = 'datasetOCR/',
                                                        h5 = 'mark1_matocr.h5')