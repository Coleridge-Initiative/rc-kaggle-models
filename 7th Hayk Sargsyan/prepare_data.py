import os
import logging
import argparse
from lib.data_utils import load_data, extract_sentences, clean_text
from lib.candidate_proposer import TrainCandidateProposer, TestCandidateProposer
from lib.data_processing import TrainDataProcessor, TestDataProcessor
from lib.utils import get_title_label_mapping, load_settings

settings = load_settings('./settings.json')
os.makedirs(settings['LOGS_DIR'], exist_ok=True)
log_path = os.path.join(settings['LOGS_DIR'], 'prepare_data.log')
logging.basicConfig(filename=log_path,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
log = logging.getLogger(__name__)


def prepare_train_data():
    """Prepare training data"""

    # Step1: Read in the publications
    pub_dir = os.path.join(settings['RAW_DATA_DIR'], 'train')
    df_path = os.path.join(settings['RAW_DATA_DIR'], 'train.csv')
    train_df = load_data(df_path, pub_dir=pub_dir)
    # Extract the sentences and save them in a separate column
    log.info('Extracing the sentences from the publication texts...')
    train_df['sentences'] = train_df['text'].apply(extract_sentences)
    title_label_mapping = get_title_label_mapping(train_df)

    # Step2: Generate the candidates and group them
    candidate_proposer = TrainCandidateProposer(train_df, settings,
                                                title_label_mapping)
    candidate_groups, _ = candidate_proposer.generate_candidates()

    # Step3: Prepare data for training
    data_processor = TrainDataProcessor(train_df, candidate_groups)
    train_data = data_processor.construct_data()
    train, valid = data_processor.split_data(train_data)
    log.info(f'Full Train data length: {len(train_data)}')
    log.info(f'Train data length: {len(train)}')
    log.info(f'Valid data length: {len(valid)}')
    train_spacy, valid_spacy = data_processor.convert_to_spacy_format(train, valid)
    # # Step4: Save the data for training
    data_processor.save_data(train_spacy, settings['TRAIN_DATA_CLEAN_PATH'])
    data_processor.save_data(valid_spacy, settings['VALID_DATA_CLEAN_PATH'])


def prepare_test_data():
    """Prepare data for inference"""

    # Step1: Read in the publications
    pub_dir = os.path.join(settings['RAW_DATA_DIR'], 'test')
    df_path = os.path.join(settings['RAW_DATA_DIR'], 'sample_submission.csv')
    # pub_dir = os.path.join(settings['RAW_DATA_DIR'], 'train')
    # df_path = os.path.join(settings['RAW_DATA_DIR'], 'train.csv')
    test_df = load_data(df_path, pub_dir=pub_dir)

    # Clean the texts and store in separate columnn
    test_df['clean_text'] = test_df['text'].apply(clean_text)

    # Save the dataframe: will be used in the inference phase
    test_df.to_csv(settings['TEST_DATA_CLEAN_PATH'])

    # Extract the sentences and save them in a separate column
    log.info('Extracing the sentences from the publication texts...')
    test_df['sentences'] = test_df['text'].apply(extract_sentences)

    train_df_path = os.path.join(settings['RAW_DATA_DIR'], 'train.csv')
    train_df = load_data(train_df_path)
    title_label_mapping = get_title_label_mapping(train_df)

    # Step2: Generate the candidates and group them
    candidate_proposer = TestCandidateProposer(test_df, settings,
                                               title_label_mapping)
    candidate_groups, id_abb_def_mappings = candidate_proposer.generate_candidates()

    # Step3: Prepare data for Classification
    data_processor = TestDataProcessor(test_df, candidate_groups)
    test_data = data_processor.construct_data()
    log.info(f'Number of sentences to classify {len(test_data)}')

    # Step4: Get candidates through a simple search of "candidate dataset"
    # occurencies
    naive_preds = data_processor.naive_search()

    # Step4: Save the data and the groups
    data_processor.save_data(test_data, settings['TEST_TEXTCAT_DATA_PATH'])
    data_processor.save_data(data_processor.groups, settings['TEST_DATA_GROUP_PATH'])
    data_processor.save_data(id_abb_def_mappings, settings['TEST_ABB_DEF_MAPPINGS_PATH'])
    data_processor.save_data(naive_preds, settings['TEST_NAIVE_PREDICTIONS_DATA_PATH'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true')
    group.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.train:
        log.info('Preparing data for training')
        # Create the necessary directoris if missing
        os.makedirs(settings['TRAIN_DATA_CLEAN_DIR'], exist_ok=True)
        os.makedirs(settings['MODEL_CHECKPOINT_DIR'], exist_ok=True)
        # Prepare the train data
        prepare_train_data()
        log.info('Finished data preparation')
    elif args.test:
        log.info(f'Preparing data for inference')
        # Create the necessary directoris if missing
        os.makedirs(settings['TEST_DATA_CLEAN_DIR'], exist_ok=True)
        prepare_test_data()
        log.info('Finished data preparation')
