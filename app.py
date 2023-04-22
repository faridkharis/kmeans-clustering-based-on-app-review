from flask import Flask, render_template, url_for, request, send_file, make_response
from google_play_scraper import app, Sort, reviews
import pandas as pd
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
import gensim
from gensim import corpora

app = Flask(__name__)


class DataStore():
    df_data = None
    df_clean = None
    document = None


data = DataStore()


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route('/download-data')
def download_data():
    df_data = data.df_data
    resp = make_response(df_data.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=Data Ulasan.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


@app.route('/download-data-clean')
def download_data_clean():
    df_clean = data.df_clean
    resp = make_response(df_clean.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=Data Ulasan Preprocessed.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


@app.route("/guide")
def guide():
    return render_template("guide.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/scraping", methods=["POST", "GET"])
def scraping():
    if request.method == 'POST':
        appId = request.form['app-id']
        reviewCount = request.form['review-count']

        result, continuation_token = reviews(
            app_id=appId,
            lang='id',
            country='id',
            sort=Sort.NEWEST,
            count=int(reviewCount),
            filter_score_with=None
        )

        result, _ = reviews(
            app_id=appId,
            continuation_token=continuation_token
        )

        df_data = pd.DataFrame(result)
        df_data = df_data.drop(columns=['userName', 'userImage', 'thumbsUpCount',
                                        'reviewCreatedVersion', 'replyContent', 'repliedAt'])
        df_data.index += 1

        data.df_data = df_data

        return render_template("review.html", tables=[df_data.to_html(classes='empTable dataTable hover')])
    else:
        return render_template("scraping.html")


@app.route("/preprocessing", methods=["GET", "POST"])
def preprocessing():
    if request.method == 'POST':
        df_clean = pd.read_csv(request.files.get('csv-file'))
        df_clean = df_clean.drop(columns=['Unnamed: 0'])
        df_clean.index += 1

        # ------ Case Folding ------
        df_clean['content'] = df_clean['content'].str.lower()

        # ------ Tokenizing ---------
        def remove_tweet_special(text):
            # remove tab, new line, ans back slice
            text = text.replace('\\t', " ").replace(
                '\\n', " ").replace('\\u', " ").replace('\\', "")
            # remove non ASCII (emoticon, chinese word, .etc)
            text = text.encode('ascii', 'replace').decode('ascii')
            # remove mention, link, hashtag
            text = ' '.join(
                re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
            # remove incomplete URL
            return text.replace("http://", " ").replace("https://", " ")

        df_clean['content'] = df_clean['content'].apply(
            remove_tweet_special)

        # remove number
        def remove_number(text):
            return re.sub(r"\d+", "", text)

        df_clean['content'] = df_clean['content'].apply(remove_number)

        # remove punctuation
        def remove_punctuation(text):
            return text.translate(str.maketrans("", "", string.punctuation))

        df_clean['content'] = df_clean['content'].apply(
            remove_punctuation)

        # remove whitespace leading & trailing
        def remove_whitespace_LT(text):
            return text.strip()

        df_clean['content'] = df_clean['content'].apply(
            remove_whitespace_LT)

        # remove multiple whitespace into single whitespace
        def remove_whitespace_multiple(text):
            return re.sub('\s+', ' ', text)

        df_clean['content'] = df_clean['content'].apply(
            remove_whitespace_multiple)

        # remove single char
        def remove_single_char(text):
            return re.sub(r"\b[a-zA-Z]\b", "", text)

        df_clean['content'] = df_clean['content'].apply(
            remove_single_char)

        # NLTK word tokenize
        def word_tokenize_wrapper(text):
            return word_tokenize(text)

        df_clean['content'] = df_clean['content'].apply(
            word_tokenize_wrapper)

        # ------ Normalization ------
        normalizad_word = pd.read_excel('static/files/normalisasi.xlsx')

        normalizad_word_dict = {}

        for index, row in normalizad_word.iterrows():
            if row[0] not in normalizad_word_dict:
                normalizad_word_dict[row[0]] = row[1]

        def normalized_term(document):
            return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]

        df_clean['content'] = df_clean['content'].apply(
            normalized_term)

        # create stemmer
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        # stemmed
        def stemmed_wrapper(term):
            return stemmer.stem(term)

        term_dict = {}

        for document in df_clean['content']:
            for term in document:
                if term not in term_dict:
                    term_dict[term] = ' '

        for term in term_dict:
            term_dict[term] = stemmed_wrapper(term)

        def get_stemmed_term(document):
            return [term_dict[term] for term in document]

        df_clean['content'] = df_clean['content'].swifter.apply(
            get_stemmed_term)

        # ------ Stopword Removal ------
        # ----------------------- get stopword from NLTK stopword -------------------------------
        # get stopword indonesia
        list_stopwords = stopwords.words('indonesian')

        # ---------------------------- manually add stopword  ------------------------------------
        # append additional stopword
        list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'kalo', 'amp', 'biar',
                               'bikin', 'bilang', 'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si',
                               'tau', 'tdk', 'tuh', 'utk', 'ya', 'jd', 'jgn', 'sdh', 'aja', 'n',
                               't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt', '&amp', 'yah',
                               'bisnis', 'indonesia', 'kulina', 'aplikasi', 'bobobox', 'oy', 'stockbit',
                               'payfazz', 'fazz', 'bukuwarung', 'buku', 'warung', 'flip', 'neu',
                               'happyfresh', 'ipiring', 'nya', 'iya', 'coba', 'sih', 'bintang', 'terima',
                               'kasih', 'terimakasih', 'the', 'deh', 'in', 'bi', 'pagi', 'dong', 'moga',
                               'ya', 'banget', 'jaya', 'maju', 'bagus', 'suka', 'lumayan', 'cepat',
                               'tambah', 'tingkat', 'layan', 'ribet', 'oke', 'mantap', 'bantu', 'banget',
                               'pakai', 'terima', 'kasih', 'keren', 'mudah', 'manfaat', 'kecewa', 'tolong',
                               'gampang', 'sukses', 'admin', 'susah', 'turun', 'naikin', 'auto', 'nomor',
                               'uninstall', 'hp', 'ganti', 'jelek', 'download',
                               'bangkrut', 'capek', 'melulu', 'ya', 'hilang', 'up', 'masuk', 'hapus',
                               'lancar', 'pokok', 'alhamdulillah', 'biar', 'minimal',
                               'laku', 'aman', 'jaga', 'terimakasih', 'terima kasih', 'gagal', 'gratis',
                               'uang', 'transaksi', 'lari', 'sungguh', 'tanggap',
                               'bikin', 'nyaman', 'sulit', 'otomatis', 'butuh', 'nyata', 'nyangkut',
                               'selamat', 'nasib', 'kena', 'maaf', 'pindah', 'belah',
                               'maklum', 'kaum', 'pas', 'muncul', 'tulis', 'salah', 'ulang', 'hasil',
                               'saing', 'kompetitor', 'rilis', 'muat', 'lambat',
                               'kendala', 'pending', 'sesuai', 'nominal', 'eh', 'via', 'tangan',
                               'sedang apa', 'gunain', 'kali', 'letak', 'ngetik', 'keyboard',
                               'eja', 'buruk', 'matang', 'mulus', 'lebih baik', 'beda', 'ayo', 'tarik',
                               'tinggal', 'kaya', 'turunin', 'kasi', 'langsung', 'kesini',
                               'share', 'whatsapp', 'galeri', 'paham', 'faktor', 'format', 'mohon',
                               'kirim', 'efektif', 'nama', 'barang', 'potong', 'angka',
                               'cantik', 'tera', 'jam', 'kadaluarsa', 'proses', 'lapor', 'sistem',
                               'buka', 'tutup', 'pilih', 'nomor', 'kartu', 'alih',
                               'bayar', 'mobile', 'layar', 'mantap', 'mantul', 'terus', 'bot'])

        # convert list to dictionary
        list_stopwords = set(list_stopwords)

        # remove stopword pada list token
        def stopwords_removal(words):
            return [word for word in words if word not in list_stopwords]

        df_clean['content'] = df_clean['content'].apply(
            stopwords_removal)

        df_clean2 = df_clean[df_clean.astype(str)['content'] != '[]']

        data.df_clean = df_clean2

        return render_template("clean.html", tables=[df_clean.to_html(classes='empTable dataTable hover')])
    return render_template("preprocessing.html")


@app.route("/clustering", methods=["GET", "POST"])
def clustering():
    if request.method == 'POST':
        df_preprocessed = pd.read_csv(request.files.get('csv-file'))
        df_preprocessed = df_preprocessed.drop(columns=['Unnamed: 0'])
        df_preprocessed.index += 1

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_extraction.text import TfidfTransformer
        vectorizer = TfidfVectorizer()

        vector_data = vectorizer.fit_transform(df_preprocessed['content'])

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=4, n_init="auto",
                        random_state=0).fit(vector_data)
        result = kmeans.labels_

        topics = []
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        for i in range(4):
            for j in order_centroids[i, :2]:
                topics.append(terms[j])

        df_preprocessed['topic'] = result

        # for i in range(6):
        #     df_preprocessed['topic'] = df_preprocessed['topic'].replace(
        #         [i], topics[i*2]+' '+topics[i*2+1])

        total_review = len(df_preprocessed.index)
        negative_review = len(df_preprocessed[(df_preprocessed['score'] <= 2)])
        neutral_review = len(df_preprocessed[(df_preprocessed['score'] == 3)])
        positive_review = len(df_preprocessed[(df_preprocessed['score'] >= 4)])

        topic_1_negative = len(
            df_preprocessed[(df_preprocessed['score'] <= 2) & (df_preprocessed['topic'] == 0)])
        topic_2_negative = len(
            df_preprocessed[(df_preprocessed['score'] <= 2) & (df_preprocessed['topic'] == 1)])
        topic_3_negative = len(
            df_preprocessed[(df_preprocessed['score'] <= 2) & (df_preprocessed['topic'] == 2)])
        topic_4_negative = len(
            df_preprocessed[(df_preprocessed['score'] <= 2) & (df_preprocessed['topic'] == 3)])

        topic_1_neutral = len(
            df_preprocessed[(df_preprocessed['score'] == 3) & (df_preprocessed['topic'] == 0)])
        topic_2_neutral = len(
            df_preprocessed[(df_preprocessed['score'] == 3) & (df_preprocessed['topic'] == 1)])
        topic_3_neutral = len(
            df_preprocessed[(df_preprocessed['score'] == 3) & (df_preprocessed['topic'] == 2)])
        topic_4_neutral = len(
            df_preprocessed[(df_preprocessed['score'] == 3) & (df_preprocessed['topic'] == 3)])

        topic_1_positive = len(
            df_preprocessed[(df_preprocessed['score'] >= 4) & (df_preprocessed['topic'] == 0)])
        topic_2_positive = len(
            df_preprocessed[(df_preprocessed['score'] >= 4) & (df_preprocessed['topic'] == 1)])
        topic_3_positive = len(
            df_preprocessed[(df_preprocessed['score'] >= 4) & (df_preprocessed['topic'] == 2)])
        topic_4_positive = len(
            df_preprocessed[(df_preprocessed['score'] >= 4) & (df_preprocessed['topic'] == 3)])

        negative_arr = [topic_1_negative, topic_2_negative,
                        topic_3_negative, topic_4_negative]
        neutral_arr = [topic_1_neutral, topic_2_neutral,
                       topic_3_neutral, topic_4_neutral]
        positive_arr = [topic_1_positive, topic_2_positive,
                        topic_3_positive, topic_4_positive]

        most_topic_perc = round(max(negative_arr)/sum(negative_arr)*100, 2)
        most_topic_index = negative_arr.index(max(negative_arr))

        return render_template(
            "clustering-result.html",
            tables=[df_preprocessed.to_html(
                classes='empTable dataTable hover')],
            topics=topics,
            total_review=total_review,
            positive_review=positive_review,
            neutral_review=neutral_review,
            negative_review=negative_review,
            negative_arr=negative_arr,
            neutral_arr=neutral_arr,
            positive_arr=positive_arr,
            most_topic_perc=most_topic_perc,
            most_topic_index=most_topic_index
        )
    else:
        return render_template("clustering.html")


@app.route("/result")
def result():
    return render_template("clustering-result.html")


if __name__ == "__main__":
    app.run(debug=True)
