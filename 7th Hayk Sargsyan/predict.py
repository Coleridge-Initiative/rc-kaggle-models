import os
import logging
import spacy
import pandas as pd
from lib.utils import load_settings, load_test_data
from lib.prediction_processor import PredictionProcessor
from lib.data_utils import clean_text, clean_label

settings = load_settings('./settings.json')
log_path = os.path.join(settings['LOGS_DIR'], 'predict.log')
logging.basicConfig(filename=log_path,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    filemode='w',
                    level=logging.INFO)
log = logging.getLogger(__name__)


def textcat_predict(data, model_path):
    spacy.prefer_gpu()
    nlp = spacy.load(model_path)
    predictions = []
    for doc, annts in nlp.pipe(data, as_tuples=True, batch_size=150):
        predictions.append((doc.text, {
            'Id': annts['Id'],
            'candidate': annts['candidate'],
            'score': doc.cats['positive']
        }))
    return predictions


def main():
    # Step1: Load the preprocessed data
    df, textcat_data, groups, abb_def_mappings, naive_preds = load_test_data(settings)

    # Step2: Get the predictions from the textcat model
    model_path = os.path.join(settings['MODEL_CHECKPOINT_DIR'], 'model-best')
    tc_preds = textcat_predict(textcat_data, model_path)

    # Start postprocessing the predictions
    processor = PredictionProcessor(tc_preds, naive_preds, df,
                                    groups, abb_def_mappings)
    # Step3: Remove the sentences from the predictions that contribute to the
    # naive search
    processor.remove_naive_preds_sentences()

    # Step4: Filter out obviously wrong candidates from the naive_preds
    # and match the rest to their respective groups through the dataset class
    processor.group_naive()

    # Step5: Filter predictions based on the candidate scores
    processor.filter_preds()

    # Step6: Group non-abbreviation candidates
    processor.group_long_candidates()

    # Step6: Group abbreviations
    processor.group_abbreviations()

    # Step8: Group/Discard remaining candidates
    processor.group_remaining()

    # Step9: Filter explicitly non-dataset groups, like organizations, etc.
    processor.filter_groups()

    # Step10: Construct final labels per group by searching in the cleaned text,
    # discarding the ones that appear much frequently in the full texts then
    # were found in the sentences that contained 'data', and finally by
    # sorting the labels according to the number of occurencies
    final_group_labels = processor.construct_final_labels()

    # Search for the final labels, restricting to the most common form
    # per group, abbreviation and maximum one another varriant
    id_list = []
    lables_list = []
    for index, row in df.iterrows():
        sample_clean_text = row['clean_text']
        row_id = row['Id']
        cleaned_labels = []
        for group in final_group_labels:
            abbreviation = final_group_labels[group]['abbreviation']
            most_common = final_group_labels[group]['most_common']
            remaining_labels = final_group_labels[group]['remaining']
            if abbreviation:
                if f' {clean_text(abbreviation)} ' in sample_clean_text:
                    cleaned_labels.append(abbreviation)
            if most_common:
                if f' {clean_text(most_common)} ' in sample_clean_text:
                    cleaned_labels.append(most_common)
            for remaining_label in remaining_labels:
                if f' {clean_text(remaining_label)} ' in sample_clean_text:
                    cleaned_labels.append(remaining_label)
                    break
        cleaned_labels = [clean_label(x) for x in cleaned_labels]
        cleaned_labels = set(cleaned_labels)
        lables_list.append('|'.join(cleaned_labels))
        id_list.append(row_id)

    # Prepare the submission dataframe and save it
    os.makedirs(settings['SUBMISSION_DIR'], exist_ok=True)
    submission = pd.DataFrame()
    submission['Id'] = id_list
    submission['PredictionString'] = lables_list
    sub_path = os.path.join(settings['SUBMISSION_DIR'], 'submission.csv')
    submission.to_csv(sub_path, index=None)


if __name__ == '__main__':
    main()
