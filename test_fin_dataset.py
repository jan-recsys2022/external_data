"""test_fin_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.utils import pad_sequences
from pickle import load as pickle_load
from numpy import newaxis
from gc import collect
import os

MAX_SEQ_LENGTH = 100

# TODO(test_fin_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(test_fin_dataset): BibTeX citation
_CITATION = """
"""


class TestFinDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for test_fin_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(test_fin_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'sessions': tfds.features.Tensor(shape=(1,MAX_SEQ_LENGTH), dtype=tf.int32),
            'purchases': tf.int32,
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('sessions', 'purchases'),  # Set to `None` to disable
        citation=_CITATION,
        disable_shuffling=True, 
    )

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Register into https://example.org/login to get the data. Place the `data.zip`
  file in the `manual_dir/`.
  """
    
  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    print("INFO: split_generators")
    """Returns SplitGenerators."""
    # data_path is a pathlib-like `Path('<manual_dir>/data.zip')`
    archive_path = dl_manager.manual_dir / 'final.zip'
    # Extract the manually downloaded `data.zip`
    extracted_path = dl_manager.extract(archive_path)

    # TODO(my_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    self.X = None
    return {
        'train': self._generate_examples(
          sessions_path=extracted_path / 'X_fin.pkl'
      )
    }

  def preprocess_seqs(self, products_ex):
    # input must be array of lists
    return tf.keras.utils.pad_sequences(
    products_ex,
    maxlen=MAX_SEQ_LENGTH,
    #dtype='int32',
    padding='post',
    #truncating='pre',
    value=0.0
    )

  def _generate_examples(self, sessions_path):
    print("INFO: _generate_examples")
    """Yields examples."""
    # TODO(my_dataset): Yields (key, example) tuples from the dataset
    if self.X is not None:
        del self.X
        collect()
    print(os.path.abspath(os.getcwd()))
    with open(sessions_path, 'rb') as handle_X:
        self.X = pickle_load(handle_X) 
        self.X = self.preprocess_seqs(self.X["item_id_new"])
        for i in range(0, self.X.shape[0]):
            yield i, {
                        "sessions": self.X[i,newaxis],
                        "purchases": 0
                     }
