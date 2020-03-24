# Modified from: https://www.kaggle.com/gauravs90/keras-bert-toxic-model-bert-fine-tuning-with-keras

from keras.callbacks import ModelCheckpoint

import keras as keras
from keras.layers import Input, concatenate

from keras_bert import load_trained_model_from_checkpoint, load_vocabulary
from keras_bert import Tokenizer
from keras_bert import AdamWarmup, calc_train_steps
from keras.callbacks import Callback, TensorBoard
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import os
import pickle
import numpy as np
import argparse
import time

TIME_START = time.time()

SEED = 2321598
RESULTS_FILE = 'results_fullcycle.csv'

for SEED in [SEED]:
    ALL_SECTIONS = ['Kotimaa', 'Talous', 'Urheilu']


    def get_arguments():
        """
        Argument parser
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--neurons", action="store", dest="neurons", required=True, help="Number of neurons per layer." )
        parser.add_argument("--bert_data_path", action="store", dest="bert_data_path", required=True, help="Path containing BERT data files.")
        parser.add_argument("--features_file", action="store", dest="features_file", required=True, help="Path for the input data (pickle), similar as exampledata.pickle")
        parser.add_argument("--use_sections", action='store', dest='use_sections', required=False, nargs='*', default=False, help="Use news article sections as additional input for the model.")
        parser.add_argument("--epochs", action="store", dest='epochs', required=False, default=10, help="Number of training epochs")
        parser.add_argument("--use_lemmatized_title", action="store_true", dest='use_lemmatized_title', required=False, help="Use titles in lemmatized format.")
        parser.add_argument('--use_temporal_features', action='store_true', dest='use_temporal_features', required=False, help="Use publishing time as a feature")
        parser.add_argument('--use_title_features', action='store', dest='use_title_features', required=False, choices=['binary', 'cont'], default=False)
        parser.add_argument("--hidden_layers", action="store", dest='hidden_layers', required=False, default=1, help="Number of custom hidden layers.")
        parser.add_argument("--calculate_response", action="store_true", required=False, default=False, help="Defines whether to calculate response variable or use the precalculated one on the input data.")
        parser.add_argument("--no_premium", action="store_true", required=False, default=False, help="Ignore content behind a paywall.")
        parser.add_argument("--prem_free_separate", action="store_true", required=False, default=False, help="Calculate response variable separately for free and premium content.")
        parser.add_argument("--model_filename", action="store_true", required=False, default="model.h5", help="Filename where to save the trained model.")

        return parser.parse_args()


    def check_section_input(opts):
        """
        To convert the section input.
        If the flag is not used at all, boolean False is stored to it and nothing will be done
        If the flag is used with no parameters, it will at first contain an empty list which is converted to True - all sections will be used
        If the flag us used with an integer, we will use that many most usual sections
        If the flag is used with string variables, they are assumed to be section names. We check whether the input is correct
        """

        if isinstance(opts.use_sections, list):
            if len(opts.use_sections) == 0:
                opts.use_sections = True
                return opts
            elif len(opts.use_sections) == 1:
                if opts.use_sections[0].isdigit():
                    opts.use_sections = int(opts.use_sections[0])
                    return opts
            for section in opts.use_sections:
                if section not in ALL_SECTIONS:
                    msg = "Input section '{}' does not exist. The sections available are: {}.".format(section, ALL_SECTIONS)
                    raise ValueError(msg)
        return opts


    def save_results(opts, res, results_file=None):
        """
        To save results line by line to a file.
        """
        if results_file is None:
            results_file = 'neural_results.csv'

        train_loss, train_acc, val_loss, val_acc, val_cmat, test_loss, test_acc, test_cmat = res
        val_cmat = str([list(val_cmat[i]) for i in range(val_cmat.shape[1])])
        test_cmat = str([list(test_cmat[i]) for i in range(test_cmat.shape[1])])

        res_str = '{};{};{};{};{};{};{};{};{}\n'
        res_str = res_str.format(opts, train_loss, train_acc, val_loss, val_acc, val_cmat, test_loss,
                                 test_acc, test_cmat)
        print("saving results {}".format(res_str))

        with open(results_file, 'a') as fp:
            fp.write(res_str)


    def initialise_results_file(results_file=None):
        """Create results file and write headers to it if it does not already exist"""
        if results_file is None:
            results_file = 'neural_results.csv'
        headers = 'opts;train_loss;train_acc;val_loss;val_acc;val_cmat;test_loss;test_acc;test_cmat\n'
        if not os.path.isfile(results_file):
            with open(results_file, 'w') as fp:
                fp.write(headers)


    def parse_title(opts):
        if opts.use_lemmatized_title:
            return 'orig_string_lemmatized'
        else:
            return 'orig_string'


    def encode_time(arr, max_val=None):
        if max_val is None:
            max_val = arr.max()
            max_val = len(np.arange(max_val))
        ar1 = np.sin(2 * np.pi * arr / max_val)
        ar2 = np.cos(2 * np.pi * arr / max_val)
        return np.column_stack((ar1, ar2))


    opts = get_arguments()
    print('########################################')
    print(opts)
    print('########################################')
    opts = check_section_input(opts)
    # print("Section option is : {} and type {}".format(opts.use_sections, type(opts.use_sections)))
    NUM_NEURONS = int(opts.neurons)
    HIDDEN_LAYERS = int(opts.hidden_layers)
    RESPONSE_VARIABLE='multiclass_response'
    TITLE_VERSION = parse_title(opts)
    TITLE_FEATURES_VARIABLE = 'title_features'
    title_feature_names = ['n_words', 'n_chars_orig', 'n_chars_lemmas', 'mean_word_len', 'starts_with_identifier',
                           'n_sents_naive', 'n_sents_title', 'has_quotation', 'has_local_city_name', 'n_question', 'n_excl', 'n_dash',
                           'n_colon', 'n_comma', 'n_dot', 'n_semicolon', 'n_loc', 'n_ordinal', 'n_gpe', 'n_org', 'n_fac',
                           'n_norp', 'n_person', 'n_cardinal', 'n_date', 'n_work_of_art', 'n_quantity', 'n_product',
                           'n_money', 'n_time', 'n_event', 'n_percent', 'n_language', 'wordclass_N', 'wordclass_V',
                           'wordclass_C', 'wordclass_A', 'wordclass_Symb', 'wordclass_Punct', 'wordclass_Pron',
                           'wordclass_Num', 'wordclass_Interj', 'wordclass_Foreign', 'wordclass_Adv', 'wordclass_Adp']


    if opts.use_title_features == 'binary':
        TITLE_FEATURES_VARIABLE = 'title_features_binarised'
        title_feature_names = ['n_words', 'n_dash', 'n_colon', 'n_comma', 'n_loc', 'n_ordinal', 'n_gpe', 'n_org',
                               'n_fac', 'n_norp', 'n_person', 'n_cardinal', 'n_date', 'n_work_of_art', 'n_quantity',
                               'n_product', 'n_money', 'n_time', 'n_event', 'n_percent', 'n_language', 'wordclass_N',
                               'wordclass_V', 'wordclass_C', 'wordclass_A', 'wordclass_Symb', 'wordclass_Punct',
                               'wordclass_Pron', 'wordclass_Num', 'wordclass_Interj', 'wordclass_Foreign',
                               'wordclass_Adv', 'wordclass_Adp', 'starts_with_identifier', 'has_quotation', 'has_local_city_name']

    initialise_results_file(RESULTS_FILE)

    SEQUENCE_LEN = 64
    BATCH_SIZE = 128
    EPOCHS = int(opts.epochs)
    LR = 1e-4

    # Language model related stuff
    pretrained_path = opts.bert_data_path
    ckpt_name = 'bert_model.ckpt'
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    checkpoint_path = os.path.join(pretrained_path, ckpt_name)
    vocab_path = os.path.join(pretrained_path, 'vocab.txt')

    # Dataset location
    FEATURES_FILE = opts.features_file


    # Load dataset and combine news sections
    with open(FEATURES_FILE, 'rb') as f:
        features = pickle.load(f)

    # Initialize tokenizer
    token_dict = load_vocabulary(vocab_path)
    tokenizer = Tokenizer(token_dict)

    # collect blacklisted sections, method depends on which use_section option is used
    BLACKLISTED_SECTIONS=['Urheilu']

    if isinstance(opts.use_sections, list):
        for section in ALL_SECTIONS:
            if section not in opts.use_sections:
                BLACKLISTED_SECTIONS.append(section)
    elif isinstance(opts.use_sections, int) & (not isinstance(opts.use_sections, bool)):
        sections = features.parsed_news_section
        sections = list(sections.value_counts().head(opts.use_sections).index)
        for section in ALL_SECTIONS:
            if section not in sections:
                BLACKLISTED_SECTIONS.append(section)

    print("Blacklisted sections are {}".format(BLACKLISTED_SECTIONS))

    # harvest the necessary data from the dataset
    titles = []
    title_targets = []
    title_features = []
    sections = []
    days_of_week = []
    hours_of_day = []
    premiums = []
    clicks = []
    read_p = []

    for key in features.index:
        section = features.loc[key]['parsed_news_section']
        if (section not in BLACKLISTED_SECTIONS) & (section in ALL_SECTIONS):
            title = features[TITLE_FEATURES_VARIABLE][key][TITLE_VERSION]
            premium = features.loc[key]['content_info']['access']

            title_target = features[RESPONSE_VARIABLE][key]

            if title is not None and title_target is not None:
                titles.append(title)
                title_targets.append(title_target)

                tf = []
                for feature_name in title_feature_names:
                    try:
                        val = float(features[TITLE_FEATURES_VARIABLE][key][feature_name])
                        tf.append(val)
                    except KeyError:
                        pass
                title_features.append(tf)
                sections.append(section)
                premiums.append(premium)
                days_of_week.append(features.loc[key]['temporal_features']['day_of_week'])
                hours_of_day.append(features.loc[key]['temporal_features']['hour_of_day'])
                clicks.append(features.loc[key]['clicks'])
                read_p.append(features.loc[key]['read_percentage'])


    # one hot encode sections
    onehot_encoder = OneHotEncoder(sparse=False)
    sections = np.asanyarray(sections).reshape(-1, 1)
    sections = onehot_encoder.fit_transform(sections)

    # fourier transform for temporal feats
    days_of_week = encode_time(np.array(days_of_week), 7)
    hours_of_day = encode_time(np.array(hours_of_day), 24)


    def convert_data(titles, title_targets, title_features, sections, premiums, days_of_week, hours_of_day, clicks, read_p):
        """
        Converts the data to the format expected by the neural network
        """
        global tokenizer

        indices, targets, orig_titles, title_features_filt, sections_filt, premiums_filt, \
        days_of_week_filt, hours_of_day_filt, clicks_filt, read_p_filt = [], [], [], [], [], [], [], [], [], []

        for i, (t, target, tf, sect, prem, day, hour, click, readp) in enumerate(zip(titles, title_targets, title_features, sections,
                                                                       premiums, days_of_week, hours_of_day, clicks, read_p)):
            if type(t)==str and target is not None:
                if opts.no_premium:
                    if prem == 'free':
                        ids, segments = tokenizer.encode(t, max_len=SEQUENCE_LEN)
                        indices.append(ids)
                        targets.append(target)
                        orig_titles.append(t)
                        title_features_filt.append(tf)
                        sections_filt.append(sect)
                        premiums_filt.append(prem)
                        days_of_week_filt.append(day)
                        hours_of_day_filt.append(hour)
                        clicks_filt.append(click)
                        read_p_filt.append(readp)
                else:
                    ids, segments = tokenizer.encode(t, max_len=SEQUENCE_LEN)
                    indices.append(ids)
                    targets.append(target)
                    orig_titles.append(t)
                    title_features_filt.append(tf)
                    sections_filt.append(sect)
                    premiums_filt.append(prem)
                    days_of_week_filt.append(day)
                    hours_of_day_filt.append(hour)
                    clicks_filt.append(click)
                    read_p_filt.append(readp)

        indices = np.array(indices)
        targets = np.array(targets)
        orig_titles = np.array(orig_titles)
        title_features_filt = np.array(title_features_filt)
        sections_filt = np.array(sections_filt)
        premiums_filt = np.array(premiums_filt)
        days_of_week_filt = np.array(days_of_week_filt)
        hours_of_day_filt = np.array(hours_of_day_filt)
        clicks_filt = np.array(clicks_filt)
        read_p_filt = np.array(read_p_filt)

        return [indices, np.zeros_like(indices)], targets, orig_titles, title_features_filt, sections_filt, premiums_filt, days_of_week_filt, hours_of_day_filt, clicks_filt, read_p_filt


    X, y, orig_titles, title_features, sections, premiums, days_of_week, hours_of_day, clicks, read_p = convert_data(titles, title_targets, title_features, sections, premiums, days_of_week, hours_of_day, clicks, read_p)
    feats_to_include = list()
    premiums = (premiums == 'premium').astype(int)
    feats_to_include.append(premiums)


    if opts.use_title_features:
        if opts.use_title_features == 'cont':
            feats_to_include.append(title_features)
        else:
            def onehot(a):
                if len(np.unique(a)) > 2:
                    a = OneHotEncoder(sparse=False, categories='auto').fit_transform(a.reshape(-1, 1))
                return a
    
            tf = np.column_stack([onehot(title_features[:, i]) for i in range(title_features.shape[1])])
            feats_to_include.append(tf) 

    if opts.use_sections:
        dont = False
        if isinstance(opts.use_sections, list):
            if len(opts.use_sections) == 1:
                dont = True
        if not dont:
            print("Adding sections to title_features")
            section_names = [section for section in ALL_SECTIONS if section not in BLACKLISTED_SECTIONS]
            print("There are {} sections, they are are {}".format(len(section_names), section_names))
            feats_to_include.append(sections)
    else:
        print("Sections not used")

    if opts.use_temporal_features:
        print("Adding temporals to title_features")
        feats_to_include.append(days_of_week)
        feats_to_include.append(hours_of_day)

    feats_to_include = np.column_stack(feats_to_include)


    def generate_label(clicks, read_percentage, click_limits, read_limits):

        click_mask = np.digitize(clicks, click_limits)
        read_mask = np.digitize(read_percentage, read_limits)

        y = np.zeros(len(clicks), dtype=int)
        y[(click_mask == 1) | (read_mask == 1)] = -1
        y[(click_mask == 2) & (read_mask == 0)] = 1
        y[(click_mask == 0) & (read_mask == 2)] = 2
        y[(click_mask == 2) & (read_mask == 2)] = 3

        binary_y = np.zeros(len(clicks), dtype=int)
        binary_y[y == -1] = -1
        binary_y[y == 3] = 1

        return y, binary_y


    y_m = np.asarray(y)

    # combine data
    X.append(feats_to_include)
    X.append(y_m)
    X.append(orig_titles)
    X.append(clicks)
    X.append(read_p)

    # shuffle
    items = list(zip(X[0], X[1], X[2], X[3], X[4], X[5], X[6]))
    np.random.seed(SEED)
    np.random.shuffle(items)
    X = list(zip(*items))
    for i in range(len(X)):
        X[i] = np.array(X[i])

    # Split dataset into train and test
    TEST_SPLIT_SIZE = 0.15
    VAL_SPLIT_SIZE = 0.15

    val_split_index = len(X[0]) - int(len(X[0]) * (VAL_SPLIT_SIZE + TEST_SPLIT_SIZE))
    test_split_index = len(X[0]) - int(len(X[0]) * (TEST_SPLIT_SIZE))

    X_train = [X[0][:val_split_index], X[1][:val_split_index], X[2][:val_split_index]]
    y_train = X[3][:val_split_index]

    X_val = [X[0][val_split_index:test_split_index], X[1][val_split_index:test_split_index], X[2][val_split_index:test_split_index]]
    y_val = X[3][val_split_index:test_split_index]

    X_test = [X[0][test_split_index:], X[1][test_split_index:], X[2][test_split_index:]]
    y_test = X[3][test_split_index:]

    orig_titles = X[4]
    clicks = X[5]
    read_p = X[6]
    premiums = X[2][:, 0]


    if opts.calculate_response:
        if opts.prem_free_separate:
            free = np.where(premiums == 0)[0]
            prem = np.where(premiums == 1)[0]
        else:
            free = np.arange(len(premiums))
            prem = []
        y_m = np.repeat(np.nan, len(y))

        if opts.response == 'clicks':
            indsfree = [i for i in range(val_split_index) if i in free]
            limsfree = list(np.quantile(clicks[indsfree], [0.45, 0.55]).round(0).astype(int))
            y_m[free] = np.digitize(clicks[free], limsfree)

            if len(prem) > 0:
                indsprem = [i for i in range(val_split_index) if i in prem]
                limsprem = list(np.quantile(clicks[indsprem], [0.45, 0.55]).round(0).astype(int))
                y_m[prem] = np.digitize(clicks[prem], limsprem)
            y_m[y_m == 1] = -1
            y_m[y_m == 2] = 1
        elif opts.response == 'read_percentage':
            indsfree = [i for i in range(val_split_index) if i in free]
            limsfree = list(np.quantile(read_p[indsfree], [0.45, 0.55]))
            y_m[free] = np.digitize(read_p[free], limsfree)

            if len(prem) > 0:
                indsprem = [i for i in range(val_split_index) if i in prem]
                limsprem = list(np.quantile(read_p[indsprem], [0.45, 0.55]))
                y_m[prem] = np.digitize(read_p[prem], limsprem)
            y_m[y_m == 1] = -1
            y_m[y_m == 2] = 1
        else:
            indsfree = [i for i in range(val_split_index) if i in free]
            click_limits = np.quantile(clicks[indsfree], [0.45, 0.55])
            read_limits = np.quantile(read_p[indsfree], [0.45, 0.55])
            y_m[free], _ = generate_label(clicks[free], read_p[free], click_limits, read_limits)

            if len(prem) > 0:
                indsprem = [i for i in range(val_split_index) if i in prem]
                click_limits = np.quantile(clicks[indsprem], [0.45, 0.55])
                read_limits = np.quantile(read_p[indsprem], [0.45, 0.55])
                y_m[prem], _ = generate_label(clicks[prem], read_p[prem], click_limits, read_limits)

        def remove_negs(xt, yt):
            inds = np.where(yt > -1)[0]
            for i in range(len(xt)):
                xt[i] = xt[i][inds]
            yt = yt[inds]
            return xt, yt

        y_train = y_m[:val_split_index]
        X_train, y_train = remove_negs(X_train, y_train)
        X, y = remove_negs(X, y_m)

        test_split_index = len(y_train) + (len(y) - len(y_train)) // 2
        val_split_index = len(y_train)

        X_val = [X[0][val_split_index:test_split_index], X[1][val_split_index:test_split_index], X[2][val_split_index:test_split_index]]
        y_val = y[val_split_index:test_split_index]

        X_test = [X[0][test_split_index:], X[1][test_split_index:], X[2][test_split_index:]]
        y_test = y[test_split_index:]

        orig_titles = orig_titles[np.where(y_m > -1)[0]]

    print("train: len {} \n {}".format(len(y_train), pd.Series(y_train).value_counts(normalize=True)))
    print("val: len {} \n {}".format(len(y_val), pd.Series(y_val).value_counts(normalize=True)))
    print("test: len {} \n {}".format(len(y_test), pd.Series(y_test).value_counts(normalize=True)))


    # Load BERT model
    model = load_trained_model_from_checkpoint(
        config_path,
        checkpoint_path,
        training=True,
        trainable=False,
        seq_len=SEQUENCE_LEN,
    )

    # Model inputs
    input_features = Input(shape=(feats_to_include.shape[1],))
    inputs = model.inputs[:2]
    inputs.append(input_features)

    # Add custom layers (transfer learning, set only the last two layers to trainable)
    dense = model.layers[-3].output
    concat = concatenate([dense, input_features])
    dense_s = keras.layers.Dense(NUM_NEURONS, activation='tanh', kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02), trainable=True)(concat)

    if HIDDEN_LAYERS > 1:
        for hid_l in range(HIDDEN_LAYERS):
            dense_s = keras.layers.Dense(int(NUM_NEURONS), activation='tanh', kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02), trainable=True)(dense_s)

    outputs = keras.layers.Dense(len(np.unique(y_train)), activation='softmax', kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                                     name = 'real_output', trainable=True)(dense_s)

    decay_steps, warmup_steps = calc_train_steps(
        y_train.shape[0],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    model = keras.models.Model(inputs, outputs)

    loss_function = 'sparse_categorical_crossentropy'

    model.compile(
        AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=LR),
        loss=loss_function,
        metrics=['accuracy']
    )

    model.summary()

    class EvaluationCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch%20==0:
                y_pred = self.model.predict(X_val)
                y_pred = np.argmax(y_pred, axis=1)

                print(confusion_matrix(y_val, y_pred))


    # Train the model
    bertname = 'fi' if 'finnish' in pretrained_path else 'multi'
    if isinstance(opts.use_sections, list):
        sect = ""
        for section in opts.use_sections:
            sect += section
    else:
        sect = opts.use_sections

    tag = 'titfeat{}_{}sect_{}prems_{}premfreesep_{}bert'.format(opts.use_title_features, sect, not opts.no_premium, opts.prem_free_separate, bertname)

    model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[ModelCheckpoint(save_best_only=True, monitor='val_loss', filepath=opts.model_filename)]
        )

    print("Evaluating performance on validation and test data")
    val_loss = model.history.history['val_loss'][-1]
    val_acc = model.history.history['val_acc'][-1]
    train_loss = model.history.history['loss'][-1]
    train_acc = model.history.history['acc'][-1]

    preds = np.argmax(model.predict(X_test), axis=1)
    test_cmat = confusion_matrix(y_test, preds)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    val_cmat = confusion_matrix(y_val, np.argmax(model.predict(X_val), axis=1))

    print(test_cmat)
    save_results(opts, [train_loss, train_acc, val_loss, val_acc, val_cmat, test_loss, test_acc, test_cmat], RESULTS_FILE)

    TIME_STOP = time.time()
    print('Total time elapsed [seconds]: ' + str(TIME_STOP-TIME_START))
