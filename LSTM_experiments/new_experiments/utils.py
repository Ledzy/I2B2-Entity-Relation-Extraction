def get_train(file_path, no_dup = True, no_other=False):
    """
    format the i2b2 data into a pandas dataframe.
    notice that there are many duplicates texts in the dataset, so adjust the parameters according
    to your interests.

    parameters:
        file_path: the file's path, a string format
        no_dup: if true, the duplicate text would be removed
        no_other: if true, the samples of tag "other" should be removed

    sample usage: train_df = get_train("./training file.txt")
    return : a pd dataframe with columns: text, tag, test_info, problem_info, treatment_info
    """

    file = open(file_path)
    file = [line.strip('\n').strip('\ufeff') for line in file.readlines()]

    def format_input(df):
        targets = ['test','problem','treatment']
        for target in targets:
            df.loc[df['t1'].str.contains('\|'+target),target+'_info'] = df['t1']
            df.loc[(df['t2'].str.contains('\|'+target)) & \
                         (df[target+'_info'].isnull()),target+'_info'] = df['t2']
        df.drop(['t1','t2'],axis=1,inplace=True)
        if no_dup:
            df.drop_duplicates(['text'],inplace=True)
        if no_other: 
            df = df.loc[df.tag!='other']  #delete tag "other"
        df.index = np.arange(df.shape[0])
        return df


    train_df = pd.DataFrame(np.array([file[i::5] for i in range(4)]).T,columns=['text','t1','t2','tag'])
    train_df = format_input(train_df)
    return train_df



def clean_str(text,lower=True):
    """
    clean and format the text

    parameters:
        text: a string format text
        lower: if true, the text would be convert to lower format
    
    return: processed text
    """

    text = text.lower()
    
    replace_pair = [(r"[^A-Za-z0-9^,!.\/'+-=]"," "),(r"what's","what is "),(r"that's","that is "),(r"there's","there is "),
                   (r"it's","it is "),(r"\'s", " "),(r"\'ve", " have "),(r"can't", "can not "),(r"n't", " not "),(r"i'm", "i am "),
                   (r"\'re", " are "),(r"\'d", " would "),(r"\'ll", " will "),(r",", " "),(r"\.", " "),(r"!", " ! "),(r"\/", " "),
                   (r"\^", " ^ "),(r"\+", " + "),(r"\-", " - "),(r"\=", " = "),(r"'", " "),(r"(\d+)(k)", r"\g<1>000"),(r":", " : "),
                   (r" e g ", " eg "),(r" b g ", " bg "),(r" u s ", " american "),(r"\0s", "0"),(r" 9 11 ", "911"),(r"e - mail", "email"),
                   (r"j k", "jk"),(r"\s{2,}", " ")]
    
    for before, after in replace_pair:
        text = re.sub(before,after,text)

    return text.strip()

def load_glove(word_index):
    def get_coefs(word,*emb): return word, np.asarray(emb,dtype='float32')
    embedding = dict(get_coefs(*o.split(' ')) for o in tqdm(open('glove.840B.300d.txt')))

    emb_mean, emb_std = -0.005838459, 0.48782179
    embed_matrix = np.random.normal(emb_mean,emb_std,(max_features,emb_size))
    
    for word, i in word_index.items():
        if i >= max_features: continue
        if embedding.get(word) is not None:
            embed_matrix[i] = embedding.get(word)
    
    return embed_matrix

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))