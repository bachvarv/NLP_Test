import tensorflow_datasets as tfds
import tensorflow_text

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True, as_supervised=True)

# dataframe = tfds.as_dataframe(examples['train'].take(-1), metadata)


