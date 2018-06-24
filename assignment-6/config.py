
FILE_NAME = "spine_data.csv"
SPINE_FEATURES = ["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "degree_spondylolisthesis", "pelvic_radius", "degree_spondylolisthesis.1"]
TARGET_COLUMN_NAME = "class"

PERCENT_TEST_SIZE = 40
HIDDEN_LAYER_NEURON_SIZES = (5)
#ACTIVATION_FUNCTION = "identity"
ACTIVATION_FUNCTION = "relu"
SOLVER_OPTIMIZATION = "adam"
MAXIMUM_ITERATION = 1500
RANDOM_STATE = 1

# List of Python standard encodings (https://docs.python.org/3/library/codecs.html#standard-encodings))
ENCODING_LATIN1="latin1"