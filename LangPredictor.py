import math

class Baseline:
    def __init__(self, sents):
        # Count frequency of each tag, and each word,tag pair
        word_tag_freqs = {}
        tag_freqs = {}
        for sent in sents:
            for token in sent:
                word,tag = token
                if word not in word_tag_freqs:
                    word_tag_freqs[word] = {}
                word_tag_freqs[word][tag] = word_tag_freqs[word].get(tag, 0) + 1
                tag_freqs[tag] = tag_freqs.get(tag, 0) + 1

        # Find the most frequent tag for each word
        self.mft_for_word = {}
        for word in word_tag_freqs:
            self.mft_for_word[word] \
                = sorted(word_tag_freqs[word],
                         key=lambda x:word_tag_freqs[word][x],
                         reverse=True)[0]

        # Find the most frequent tag
        self.mft = sorted(tag_freqs, 
                          key=lambda x : tag_freqs[x], reverse=True)[0]

    def tag(self, sent):
        # Tag each token as the most frequent tag for that word; if we
        # haven't seen a word in the training data, just tag it as the
        # most frequent tag.
        return [self.mft_for_word.get(word, self.mft) for word in sent]

class Hmm:
    def __init__(self, sents):
        # Count frequency of each tag, and each word,tag pair
        self.word_tag_freqs = {}
        self.tag_freqs = {}
        self.t_probs = {}
        for sent in sents:
            n = 0
            for token in sent:
                word,tag = token

                if n == 0:
                    p_tag = "<S>"
                else:
                    p_word,p_tag = sent[n-1]

                if tag not in self.t_probs:
                    self.t_probs[tag] = {}
                self.t_probs[tag][p_tag] = self.t_probs[tag].get(p_tag,0) + 1

                if word not in self.word_tag_freqs:
                    self.word_tag_freqs[word] = {}
                self.word_tag_freqs[word][tag] = self.word_tag_freqs[word].get(tag, 0) + 1
                self.tag_freqs[tag] = self.tag_freqs.get(tag, 0) + 1
                n += 1


        for x in self.t_probs:
            count_t = 0
            for y in self.t_probs[x]:
                count_t += self.t_probs[x][y]
            for y in self.t_probs[x]:
                self.t_probs[x][y] = (self.t_probs[x][y] * 1.0) / (count_t * 1.0)
            for y in self.tag_freqs:
                if y not in self.t_probs[x]:
                    self.t_probs[x][y] = self.t_probs[x].get(y,0.0)
        
        #for x in self.t_probs:
        #    print x
        #    for y in self.t_probs[x]:
        #       print y,self.t_probs[x][y]
        
        count_t = 0
        for x in self.tag_freqs:
            count_t += self.tag_freqs[x]
        for x in self.tag_freqs:
            self.tag_freqs[x] = (self.tag_freqs[x] * 1.0) / (count_t * 1.0)

        for x in self.word_tag_freqs:
            count_f = 0
            for y in self.word_tag_freqs[x]:
                count_f += self.word_tag_freqs[x][y]
            for y in self.word_tag_freqs[x]:
                self.word_tag_freqs[x][y] = (self.word_tag_freqs[x][y] * 1.0) / (count_f * 1.0)


    def tag(self, sent):
        # Tag each token as the most frequent tag for that word; if we
        # haven't seen a word in the training data, just tag it as the
        # most frequent tag.

        table = []
        counter = 0
        for x in sent:
            word = []


            if x not in self.word_tag_freqs:
                    self.word_tag_freqs[x] = self.tag_freqs

            for y in self.word_tag_freqs[x]:
                if counter == 0:
                    word.append([y,"<S>",math.exp(math.log(self.word_tag_freqs[x][y]) )])
                else:
                    candidates = []
                    for z in self.word_tag_freqs[sent[counter-1]]:
                        if self.word_tag_freqs[x][y] == 0 or self.t_probs[y][z] == 0:
                            candidates.append([y,z,0.0])
                        else:
                            carryover = 0 
                            for w in range(len(table[counter-1])):
                                if table[counter-1][w][0] == z:
                                    carryover = table[counter-1][w][2]
                            if carryover == 0:
                                candidates.append([y,z,0.0])
                            else:
                                candidates.append([y,z,math.exp(math.log(self.word_tag_freqs[x][y]) + math.log(self.t_probs[y][z]) + math.log(carryover))])
                    highest = ["","", 0.0]
                    for z in range(len(candidates)):
                        if candidates[z][2] > highest[2]:
                            highest = candidates[z]
                    word.append(highest)
            table.append(word)
            counter += 1

        prediction = []

        for x in reversed(range(len(table))):
            if len(prediction) == 0:
                prediction.append(table[x][0])
            else:
                for y in table[x]:
                    if y[0] == prediction[len(table) - x - 2][1]:
                        prediction.append(y)
        tags = []

        for x in reversed(range(len(prediction))):
            tags.append(prediction[x][0])
        return tags
        
def print_tagged_sent(sent, tags,writer):
    for x in zip(sent, tags):
        writer.write(x[0] + '\t' + x[1]+"\n")
    writer.write("\n")


# Read a corpus with or without POS tags and return a list of
# sentences. If tags=False, return a list of sentences, where each
# sentence is a list of tokens. If tags=True, return a list of
# sentences, where each sentence is a list of POS tag,token tuples.
def sents_from_file(infile, tags=False):
    sents = []
    curr_tokens = []
    for line in infile:
        if line.strip():
            line = [x.strip() for x in line.split('\t')]
            if tags:
                token = (line[0],line[1])
            else:
                token = line[0]
            curr_tokens.append(token)
        else:
            sents.append(curr_tokens)
            curr_tokens = []
    return sents

if __name__ == '__main__':
    import sys
    train_sents = sents_from_file(open(sys.argv[1]), tags=True)
    test_sents = sents_from_file(open(sys.argv[2]), tags=False)
    # mode will be one of 'baseline' or 'hmm'; for this starter code
    # it's ignored
    mode = sys.argv[3]

    if mode == "baseline":
        tagger = Baseline(train_sents)
    elif mode == "hmm":
        tagger = Hmm(train_sents)
    else:
        print "please select either baseline, or hmm as your part of speech tagging method"
        sys.exit()

    output = open(sys.argv[2].split(".")[0] + "." + sys.argv[2].split(".")[1] + ".out." + mode + ".txt" ,"w")
    for s in test_sents:
        tags = {}
        tags = tagger.tag(s)
        print_tagged_sent(s,tags,output)
    output.close()
