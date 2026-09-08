"""Weapon detection model training modules organized by model iteration."""

import sys

from training.model_3 import third_model_dataset
sys.modules['training.third_model_dataset'] = third_model_dataset
from training.model_4 import fourth_model_dataset
sys.modules['training.fourth_model_dataset'] = fourth_model_dataset
from training.model_5 import fifth_model_dataset
sys.modules['training.fifth_model_dataset'] = fifth_model_dataset
from training.model_6 import sixth_model_dataset
sys.modules['training.sixth_model_dataset'] = sixth_model_dataset
from training.model_7 import seventh_model_dataset
sys.modules['training.seventh_model_dataset'] = seventh_model_dataset
from training.model_8 import eighth_model_dataset
sys.modules['training.eighth_model_dataset'] = eighth_model_dataset
from training.model_9 import ninth_model_dataset
sys.modules['training.ninth_model_dataset'] = ninth_model_dataset
