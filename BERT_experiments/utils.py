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



# Thanks to https://www.kaggle.com/httpwwwfszyc/bert-in-keras-taming
def convert_lines(example, max_seq_length,tokenizer):
    """convert the given texts to BERT format sequences

    parameters:
        example: a list of text string of a pandas series
        max_seq_length: pad the text to max_seq_length
        tokenizer: bert tokenizer

    sample usage:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        sequences = convert_lines(train_df["text"].fillna("DUMMY_VALUE"), 100 ,tokenizer)

    return: formatted sequence
    """
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)