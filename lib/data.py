import re
from collections import Counter
import random
import numpy as np
import csv

class Data_Processor:
    def __init__(self, rare_word_threshold=5, sampling_threshold=1e-5, sampling_text=False):
        self.rare_word_threshold = rare_word_threshold
        self.sampling_threshold = sampling_threshold
        self.sampling_text = sampling_text
        self.int2word = {}
        self.word2int = {}

    def label_special_token(self, text):
        # sentc = re.sub('<', ' <LEFT_TITLE_QUOTE> ', sentc)   # Title
        # sentc = re.sub('>', ' <RIGHT_TITLE_QUOTE> ', sentc)  # Title

        text = re.sub('https?://[-a-zA-Z/.\d_#]*', ' ', text) # http
        text = re.sub('www.[-a-zA-Z/.\d_#]*', ' ', text)  # http

        text = re.sub('[.a-zA-Z ]*[@＠][.a-zA-Z ]*', ' ', text) # email

        # sentc = re.sub('\d*\.', ' <NINDEX> ', sentc) # 12.
        # sentc = re.sub('\(\w\)', ' <AINDEX> ', sentc) # (a)
        #
        # sentc = re.sub('[。]', ' <PERIOD> ', sentc)
        # sentc = re.sub('[，,]', ' <COMMA> ', sentc)
        # sentc = re.sub('"', ' <QUOTATION_MARK> ', sentc)
        # sentc = re.sub('[;；]', ' <SEMICOLON> ', sentc)
        # sentc = re.sub('[!！]', ' <EXCLAMATION_MARK> ', sentc)
        # sentc = re.sub('[?？]', ' <QUESTION_MARK> ', sentc)
        # sentc = re.sub('[%％]', ' <PERCENTAGE> ', sentc)
        # sentc = re.sub('[（(]', ' <LEFT_PAREN> ', sentc)
        # sentc = re.sub('[)）]', ' <RIGHT_PAREN> ', sentc)
        # sentc = re.sub('《', ' <LEFT_B_PAREN> ', sentc)
        # sentc = re.sub('》', ' <RIGHT_B_PAREN> ', sentc)
        # sentc = re.sub('[「『]', ' <LEFT_B_QUOTE> ', sentc)
        # sentc = re.sub('[」』]', ' <RIGHT_B_QUOTE> ', sentc)
        # sentc = re.sub('-', ' <HYPHENS> ', sentc)
        # sentc = re.sub('、', ' <PAUSE> ', sentc)
        # sentc = re.sub('/', ' <OR> ', sentc)
        # sentc = re.sub('[:：]', ' <COLON> ', sentc)
        # sentc = re.sub('[*＊]', ' <STAR> ', sentc)
        # sentc = re.sub('\n', ' <NEW_LINE> ', sentc)

        return text

    def split(self, text):
        """
        :param text: text
        :return: [word]
        """
        return [word for word in text[:200000000] if word != ' ' and word != '\n']


    def get_token_freq(self, words):
        """
        :param words: raw [word]
        :return: self.token_count = {"theater": 12, ...}
        """
        token_count =  Counter(words)
        return token_count

    def filter_out_word(self, words, token_count):
        """
        :param words: raw [word], token_count
        :use: self.rare_word_threshold
        :return: filtered [word]
        """
        filtered_words = [word for word in words if token_count[word] > self.rare_word_threshold]
        return filtered_words

    def create_lookup_table(self, token_count):
        """
        :param token_count: token_count
        :update: self.int2word, self.word2int
        :return: Bool

        :NOTE: the int is ascending sorted by the word frequency
        """
        sorted_words = sorted(token_count, key=token_count.get, reverse=True)
        self.int2word = {ii: word for ii, word in enumerate(sorted_words)}
        self.word2int = {word: ii for ii, word in self.int2word.items()}
        return True

    def sub_sampling(self, words, token_count):
        """
        :param words: filtered [word], token_count
        :return: sampled [word]
        """
        total_count = len(words)
        freqs = {word: count/total_count for word, count in token_count.items()}
        p_drops = {word: 1-(np.sqrt(self.sampling_threshold/freqs[word])) for word in words}
        train_words = [word for word in words if random.random() < (1-p_drops[word])]
        return train_words

    def word_to_int(self, words):
        """
        :param words: sampled [word]
        :use: self.word2int
        :return: [index]
        """
        return [self.word2int[word] for word in words]

    def read_txt_file(self, path):
        """
        :param path: str
        :return: return text
        """
        file = open(path, "r")
        return file.read()

    def preprocess_text(self, path):
        """
        :param path: path
        :return: sampled [index]
        """
        file = open(path, "r")
        print("reading.....")
        text = file.read()
        print("finished")
        text = self.label_special_token(text)
        raw_words = self.split(text)

        # get the count, filter, update look up table
        token_count = self.get_token_freq(raw_words)
        filtered_words = self.filter_out_word(raw_words, token_count)
        self.create_lookup_table(token_count)

        # sampling words, words to integer
        if self.sampling_text:
            sampled_words = self.sub_sampling(filtered_words, token_count)
        else:
            sampled_words = filtered_words
        train_tokens = self.word_to_int(sampled_words)

        file.close()

        return train_tokens

    def get_target(self, words, idx, window_size=5):
        """
        :param words: sampled [word] / [index]
        :param idx: int
        :param window_size: int
        :return: [target_words]
        """
        R = np.random.randint(1, window_size + 1)
        start = idx - R if (idx - R) > 0 else 0
        stop = idx + R
        target_words = words[start:idx] + words[idx + 1:stop + 1]

        return list(target_words)

    def get_batches(self, words, batch_size, window_size):
        """
        :param words: sampled [word] / [index]
        :param batch_size: int
        :param window_size: int
        :return: yield batch
        """
        n_batches = len(words) // batch_size

        # only full batches
        words = words[:n_batches * batch_size]

        for idx in range(0, len(words), batch_size):
            x, y = [], []
            batch = words[idx:idx + batch_size]
            for ii in range(len(batch)):
                batch_x = batch[ii]
                batch_y = self.get_target(batch, ii, window_size)
                y.extend(batch_y)
                x.extend([batch_x] * len(batch_y))
            yield x, y

class Data_Processor_Sentc:
    def __init__(self, rare_word_threshold=5, sampling_threshold=1e-5):
        self.rare_word_threshold = rare_word_threshold
        self.sampling_threshold = sampling_threshold
        self.int2word = {}
        self.word2int = {}

    def label_special_token(self, text):
        text = re.sub('<', '', text)   # Title
        text = re.sub('>', '', text)  # Title

        text = re.sub('https?://[-a-zA-Z/.\d_#]*', '', text) # http
        text = re.sub('www.[-a-zA-Z/.\d_#]*', '', text)  # http
        text = re.sub('[.a-zA-Z ]*[@＠][.a-zA-Z ]*', '', text) # email
        text = re.sub('[\d]', '', text)  # number
        text = re.sub('[a-zA-Z]', '', text) # alphabet
        text = re.sub('[ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ]', '', text)
        text = re.sub('[ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ]', '', text)
        text = re.sub('[🄀⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑⒒⒓⒔⒕⒖⒗⒘⒙⒚⒛]', '', text)
        text = re.sub('[１２３４５６７８９０]', '', text)
        text = re.sub('[㆒㆓㆔㆕五]', '', text)
        text = re.sub('[一二三四五六七八九十]', '', text)
        text = re.sub('[¹²³⁴⁵⁶⁷⁸⁹⁰]', '', text)
        text = re.sub('[₁₂₃₄₅₆₇₈₉₀]', '', text)
        text = re.sub('[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]', '', text)
        text = re.sub('[ⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹⅺⅻ]', '', text)
        text = re.sub('[㈠㈡㈢]㈣㈤㈥㈦㈧㈨㈩]', '', text)
        text = re.sub('[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇]', '', text)
        text = re.sub('[ΦΧΨΩανξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥζηθλμδβ]', '', text)
        text = re.sub('[.·⋅•・·．]', '', text)  # alphabet
        text = re.sub('[。°˚]', '', text)
        text = re.sub('[，,]', '', text)
        text = re.sub('["`\']', '', text)
        text = re.sub('[;；]', '', text)
        text = re.sub('[!！¡]', '', text)
        text = re.sub('[?？]', '', text)
        text = re.sub('[%％]', '', text)
        text = re.sub('[{｛﹛【《＜〈〔［[（(]', '', text)
        text = re.sub('[}｝﹜】》＞〉〕］\])）]', '', text)
        text = re.sub('[「『]', '', text)
        text = re.sub('[」』]', '', text)

        text = re.sub('[-─−―]', '', text)
        text = re.sub('、', '', text)
        text = re.sub('[/∕／]', '', text)    # "\" replace forward slash
        text = re.sub('\\\\', '', text) # "\" replace backward slash
        text = re.sub('[:：∶﹕︰]', '', text)
        text = re.sub('[*＊﹡]', '', text)
        text = re.sub('[+]', '', text)
        text = re.sub('[&]', '', text)
        text = re.sub('[＝=]', '', text)
        text = re.sub('[℃]', '', text)
        text = re.sub('[~～]', '', text)
        text = re.sub('[é]', '', text)
        text = re.sub('[α]', '', text)
        text = re.sub('[#]', '', text)
        text = re.sub('[\^]', '', text)
        text = re.sub('[〃]', '', text)
        text = re.sub('[◇◆◈◊⟐⋄⟡⎏⎐⎑⎒⟢⟣⧪⧫⧰⍚⟠⟕⟖⟗]', '', text) # diamond
        text = re.sub('[⊖⊘⊚⊛⊜⊝◉○￮◌⧂◍◎●◐◑◒◓◔◕⧃◖◗◦◯⦾⦿⊕⊗⬲❍⬤⬬⬭⬮⬯⬰⧭⧬⧳⧲]', '', text) # circle
        text = re.sub('[★✰☆✩✫✬✭✮✡⋆✢✣✤✥✦✧✪✯❂❉✱✲✳✴✵✶✷✸✹❊✻✼❆❇❈⁂⁑]', '', text) # star
        text = re.sub('[♤♡♧♢♠♥♣♦]', '', text)  # poker
        text = re.sub('[■]', '', text)
        text = re.sub('С', '', text)

        text = re.sub('[౸טﷸﮈﮨـ׳ﵨݘ]', '', text)  # very special char
        # text = re.sub('\n', ' <NEW_LINE> ', text)
        return text

    def get_token_freq_from_sentcs(self, sentcs):
        """
        :param sentcs: raw [sentc]
        :return: self.token_count = {"theater": 12, ...}
        """
        total_sentc = len(sentcs)
        break_point = 100000
        token_count = Counter()
        sub_sentcs = ''
        for index, sentc in enumerate(sentcs):
            if index % break_point == 0 or index == (total_sentc-1):
                words = [word for word in sub_sentcs if word != ' ' and word != '\n']
                t_c = Counter(words)
                token_count = token_count + t_c
                sub_sentcs = ''
                print("Progress: ", index / total_sentc * 100, "%")
            else:
                sub_sentcs = sub_sentcs + sentc
        return token_count

    def filter_out_word_from_sentcs(self, sentcs, token_count):
        """
        :param sentcs: [sentc]
        :param token_count: {'a': 12, ...}
        :return: [filtered_sentc]
        """
        for index, sentc in enumerate(sentcs):
            filtered_sentc = ''
            for word in sentc:
                if int(token_count[word]) > self.rare_word_threshold:
                    filtered_sentc = filtered_sentc + word
            sentcs[index] = filtered_sentc
        return sentcs

    def create_lookup_table(self, token_count):
        """
        :param token_count: token_count
        :update: self.int2word, self.word2int
        :return: Bool

        :NOTE: the int is ascending sorted by the word frequency
        """
        sorted_words = sorted(token_count, key=token_count.get, reverse=True)
        self.int2word = {ii: word for ii, word in enumerate(sorted_words)}
        self.word2int = {word: ii for ii, word in self.int2word.items()}
        return True

    def sub_sampling_from_sentcs(self, sentcs, token_count):
        """
        :param sentcs: [sentc]
        :param token_count: {'a': 12, ...}
        :return: [sentc]
        """
        total_count = 0
        for sentc in sentcs:
            total_count = total_count + len(sentc)

        sampled_sentcs = []
        for sentc in sentcs:
            s = ''
            freqs = {word: count / total_count for word, count in token_count.items()}
            p_drops = {word: 1 - (np.sqrt(self.sampling_threshold / freqs[word])) for word in sentc}
            for word in sentc:
                if random.random() < (1 - p_drops[word]):
                    s = s + word
            sampled_sentcs.append(s)
        return sampled_sentcs

    def word_to_int(self, words):
        """
        :param words: sampled [word]
        :use: self.word2int
        :return: [index]
        """
        return [self.word2int[word] for word in words]

    def word_to_int_from_sentc(self, sentc):
        """
        :param sentc: sentc
        :return: [index]
        """
        pass

    def read_txt_file(self, path):
        """
        :param path: str
        :return: return text
        """
        file = open(path, "r")
        return file.read()

    def write_sentcs_file(self, sentcs, path, file_name):
        """
        :param sentcs: [sentc]
        :return: Bool
        """
        full_path = path + '/' + file_name
        with open(full_path, 'w') as f:
            for sentc in sentcs:
                f.write("%s\n" % sentc)
        return True

    def read_sentcs_file(self, path, file_name):
        """
        :param path: str
        :param file_name: str
        :return: [sentc]
        """
        full_path = path + '/' + file_name
        sentcs = []
        with open(full_path, newline='') as f:
            reader = csv.reader(f, delimiter=',', quotechar='\n')
            for row in reader:
                if len(row) == 0:
                    sentcs.append('')
                else:
                    sentcs.append(row[0])
        return sentcs

    def write_token_count_file(self, token_count, path, file_name):
        """
        :param token_count: dict {'a': 1232}
        :param path: str
        :return: Bool
        """
        full_path = path + '/' + file_name
        with open(full_path, 'w') as f:
            for key, value in token_count.items():
                f.write("%s,%s\n" % (key, value))
        return True

    def read_token_count_file(self, path, file_name):
        """
        :param path: str
        :param file_name: str
        :return: token_count
        """
        full_path = path + '/' + file_name
        token_count = {}
        with open(full_path, newline='') as f:
            reader = csv.reader(f, delimiter=',', quotechar='\n')
            for row in reader:
                token_count[row[0]] = int(row[1])
        return token_count

    def write_word_int_pair_file(self, path, file_name):
        """
        :param token_count: dict {'a': 1232}
        :param path: str
        :return: Bool
        """
        full_path = path + '/' + file_name
        with open(full_path, 'w') as f:
            for key, value in self.word2int.items():
                f.write("%s,%s\n" % (key, value))
        return True

    def read_word_int_pair_file(self, path, file_name):
        """
        :param path: str
        :param file_name: str
        :return: token_count
        """
        full_path = path + '/' + file_name
        with open(full_path, newline='') as f:
            reader = csv.reader(f, delimiter=',', quotechar='\n')
            for row in reader:
                self.word2int[row[0]] = int(row[1])
                self.int2word[int(row[1])] = row[0]
        return True

    def split2token(self, sentc):
        """
        :param sentc: str
        :return: [index]
        """
        train_tokens = []
        for word in sentc:
            word_index = self.word2int[word]
            train_tokens.append(word_index)
        return train_tokens

    def split2sentc(self, text):
        """
        :param text: text
        :return: [sentc]
        """
        text = re.sub(' ', '', text)
        return re.split(r'\n', text)

    def preprocess_text(self, main_path, target_file_path, filter_rare_words=False, sampling_words=False):
        """
        :param path: path
        :return: sampled [index]
        """
        full_path = main_path + '/' + target_file_path
        file = open(full_path, "r")
        print("Reading File.....")
        text = file.read()
        print("Read File Finished!")

        # label special character
        print("Labeling...")
        text = self.label_special_token(text)
        print("Successful!")
        raw_sentcs = self.split2sentc(text)

        # get the count, filter, update look up table
        token_count = self.get_token_freq_from_sentcs(raw_sentcs)
        print("Successful to get token frequency from file.")
        print("Writing token count as CSV...")
        write_ok = self.write_token_count_file(token_count, main_path, "token_count.csv")
        if (write_ok):
            print("Successful!")

        # filter word if the word is very rare
        if filter_rare_words:
            filtered_sentcs = self.filter_out_word_from_sentcs(raw_sentcs, token_count)
        else:
            filtered_sentcs = raw_sentcs

        # update the word2int and int2word table
        self.create_lookup_table(token_count)
        print("Writing word-int pair as CSV...")
        write_ok = self.write_word_int_pair_file(main_path, "wordIntPair.csv")
        if write_ok:
            print("Successful!")

        # sampling words, words to integer
        if sampling_words:
            sampled_sentcs = self.sub_sampling_from_sentcs(filtered_sentcs, token_count)
        else:
            sampled_sentcs = filtered_sentcs

        print("Writing the sentcs as CSV...")
        write_ok = self.write_sentcs_file(sampled_sentcs, main_path, "sentcs.csv")
        if write_ok:
            print("Successful!")

        file.close()

        return sampled_sentcs, token_count

    def get_target(self, words, idx, window_size=5):
        """
        :param words: sampled [word] / [index]
        :param idx: int
        :param window_size: int
        :return: [target_words]
        """
        R = np.random.randint(1, window_size + 1)
        start = idx - R if (idx - R) > 0 else 0
        stop = idx + R
        target_words = words[start:idx] + words[idx + 1:stop + 1]

        return list(target_words)

    def get_batches(self, sentcs, batch_size, window_size, BUFFER_SIZE=100000):
        """
        :param sentcs: [sentc]
        :param batch_size: int
        :param window_size: int
        :param BUFFER_SIZE: int default =100000
        :return: yield train_index, target_index
        """
        train_tokens = []
        last_sentc_index = len(sentcs)
        for i, sentc in enumerate(sentcs):
            # split the sentence and tokenize the words into [index]
            sentc_token = self.split2token(sentc)
            train_tokens.extend(sentc_token)

            if (i+1) % BUFFER_SIZE == 0 or (i+1) == last_sentc_index:
                if i+1 == last_sentc_index: # the last training set after slicing many buffer
                    n_batches = len(train_tokens) // batch_size
                    train_tokens = train_tokens[:n_batches*batch_size] # only complete set needed
                # yield batches
                last_idx = 0
                for idx in range(0, len(train_tokens), batch_size):
                    x, y = [], []
                    batch = train_tokens[idx:idx + batch_size]
                    for ii in range(len(batch)):
                        batch_x = batch[ii]
                        batch_y = self.get_target(batch, ii, window_size)
                        y.extend(batch_y)
                        x.extend([batch_x] * len(batch_y))
                    yield x, y
                    last_idx = idx
                train_tokens = train_tokens[last_idx:] # keep the rest for next iteration, return [] if out of range



