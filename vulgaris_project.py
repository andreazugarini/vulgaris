
""" Copyright (C) 2020 Matteo Tiezzi, Andrea Zugarini - All Rights Reserved
    Data Copyright © Copyright 2020 Sapienza Università di Roma
    You may use, distribute and modify this code under the
    terms of the license, which unfortunately won't be written for another century.

"""

import requests
from lxml import etree
import pandas as pd
import logging
import string

# logger settings
logger = logging.getLogger('Poem Parser')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('parser.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

print("The data belongs to Biblioteca Italiana, http://www.bibliotecaitaliana.it/ , all the rights reserved. The data are available for academic and non-commercial use.")
print("You agree to the terms and conditions of the Biblioteca Italiana Copyright by pressing Enter...")
input("Press Enter to continue...")
print("Initializing parser...\n")

# global counter
class Counter:
    def __init__(self):
        self.id = 0  # global over all collections

    def grow(self):
        self.id += 1


# poem web page anchors and positions
class Page:
    def __init__(self, url, title, body, family="others"):
        self.url = url  # web page URL
        self.title = title  # XPath position of the page title
        self.body = body  # XPath position of the body
        self.family = family


# global dataframe wrapper
class Dataframe_Wrap:
    def __init__(self, columns):
        columns = columns
        self.df = pd.DataFrame(columns=columns)  # class dataframe

    def appending(self, data):
        self.df = self.df.append(data, ignore_index=True)

    def to_csv(self, name, sep='\t', encoding='utf-8'):
        self.df.to_csv(name, sep, encoding)


# container for a single poem
class Poem:
    def __init__(self, title, author, collection, poem_obj, gl_id: Counter, df: Dataframe_Wrap,
                 df_poem: Dataframe_Wrap, family="others", type="poetry"):
        # initialize id and increment counter
        self.global_id = gl_id.id
        gl_id.grow()
        # poem info
        self.collection = collection
        # self.poem_type = poem_type
        self.author = author
        self.title = title
        self.family = family
        # etree object containing poem
        self.poem_obj = poem_obj
        # global dataframe where to append
        self.df = df
        self.df_poem = df_poem
        self.type = type
        # begin stanza building
        self.stanzas_poem = self.build()
        self.poem = "<EOS>".join(self.stanzas_poem) + "<EOS>"  # getting all the poem in one string

    def build(self):

        poem = []  # every element is a stanza
        for item in self.poem_obj:
            # here we are at <lg> level => every element is a stanza
            # used to avoid <head> and other tags in  rime eteree and  <argument> <p>
            if item.tag != "head" and item.tag != "argument" and item.tag != "opener" and item.tag != "salute" and item.tag != "byline":
                stanzas = ""
                for elem in item:  # <l> level => it is a phrase
                    # print(elem)
                    stanzas += (''.join(elem.itertext()) + "<EOL>" if elem.tag != "head" else '').replace("\n",
                                                                                                          "").replace(
                        "\t", "")
                poem.append(stanzas)
        return poem

    def dataset_builder(self, global_id=None):
        # populating the previous/current stanza dataframe

        poem_gl = pd.Series(
            [self.global_id, self.author, self.title, self.collection, self.family, self.type, self.poem],
            index=self.df_poem.df.columns)
        self.df_poem.appending(poem_gl)


class Prose(Poem):
    def __init__(self, title, author, collection, poem_obj, gl_id: Counter, df: Dataframe_Wrap,
                 df_poem: Dataframe_Wrap, family="others", type="prose"):
        super(Prose, self).__init__(title, author, collection, poem_obj, gl_id, df, df_poem, family, type)

    def _custom_iter_text(self, etree):
        """Create text iterator.

        The iterator loops over the element and all subelements in document
        order, returning all inner text.

        """
        tag = etree.tag

        if not isinstance(tag, str) and tag is not None:
            return
        t = etree.text
        if t:
            yield ' '.join(t.split())
        for e in etree:
            avoid_flag = False
            if e.tag != "l" and e.tag != "lg" and e.tag != "closer" and e.tag != "opener":  # avoiding this tag
                if "rend" in e.attrib:
                    if e.attrib["rend"] == "block":
                        avoid_flag = True
                        yield " [...] "  # TODO added so that the previous : can be somehow removed
                if "lang" in e.attrib:
                    if e.attrib["lang"] == "lat":
                        if not avoid_flag:  # to yeld it only ones
                            yield " [...] "
                        avoid_flag = True
                if e.tag == "title" or e.tag == "emph" or e.tag == "add" or e.tag == "del" or e.tag == "hi" or e.tag == "pb":  # continue also if this tag
                    avoid_flag = True
                    yield " "
                if e.tag == "foreign":  # continue also if this tag
                    e.text = ""  # overwrite text
                    if not avoid_flag:  # to yeld it only ones
                        yield " [...] "
                        avoid_flag = True

                if avoid_flag:
                    yield from self._custom_iter_text(e)  # recursive call
                    t = e.tail
                    if t:
                        if e.tag == "title" or e.tag == "emph" or e.tag == "add" or e.tag == "del" or e.tag == "hi" or e.tag == "pb":
                            yield ' '
                        yield ' '.join(t.split())

    def build(self):
        poem = []  # every element is a paragraph
        for item in self.poem_obj:
            # here we are at <p> level => every element is a paragraph
            if item.tag != "head" and item.tag != "quote" and item.tag != "closer" and item.tag != "opener" and item.tag != "argument" and item.tag != "lg":
                # and item.tag != "argument" and item.tag != "opener" and item.tag != "salute" and item.tag != "byline":
                paragraph = (''.join(self._custom_iter_text(item))).replace("\n", " ").replace("\t", " ")
                # paragraph = (''.join(item.itertext())).replace("\n", "").replace("\t", "")
                poem.append(paragraph)
        return poem


# a certain web page can contain more poems/canti, we wrap it into a collection, having a title, author, number of poems
class Collection:
    def __init__(self, link, gl_id: Counter, df: Dataframe_Wrap, df_poem: Dataframe_Wrap, path_poem, path_title=None,
                 family="others", type="poetry"):
        # global counter id
        self._global_id = gl_id
        # collection url
        self.link = link
        # global dataframes for stanzas and text
        self.df = df
        self.df_poem = df_poem
        self.list_avoid = ["<", ">", "[", "]", "\t", "\n", "."]
        self.family = family
        self.type = type

        # get the whole page
        getter = requests.get(link)
        # extract xml tree from page
        self.tree = etree.fromstring(getter.content)
        # get the author  and title
        self.author = self.get_author()
        self.collection_title = self.get_collection_title()
        # getting title of every poem and their number
        self.poems_titles, self.poems_number = self.get_poems(path_title)
        # extracting the poem
        self.poem_list = self.build(path_poem)

    def get_author(self):
        return self.tree.xpath(
            '//teiHeader//titleStmt/author')[0].text

    def get_collection_title(self):

        return self.tree.xpath(
            '//teiHeader//titleStmt/title')[0].text.translate({ord(x): '' for x in self.list_avoid})

    def get_poems(self, path_title='(//text/body/div1//head[1])'):
        # default position of the title
        if path_title is None:  # TODO check if works
            path_title = '(//text/body/div1//head[1])'
        # handling pages with peculiar structure - page xml style denoted with Gonzaga
        if path_title == "GONZAGA":
            poems_title_obj = self.tree.xpath('//text/body/div1')
            poems_titles = [poems_title_obj[0].attrib["n"]]
            poems_number = 1
            return poems_titles, poems_number
        # handling pages with peculiar structure
        if path_title == "LUCRE":
            poems_title_obj = self.tree.xpath('//text/body/div1')
            poems_titles = [i.attrib["n"] for i in poems_title_obj]
            poems_number = len(poems_titles)
            return poems_titles, poems_number
        if path_title == "LUCRE2":
            poems_title_obj = self.tree.xpath('//text/body/div1/div2')
            poems_titles = [i.attrib["n"] for i in poems_title_obj]
            poems_number = len(poems_titles)
            return poems_titles, poems_number
        if path_title == "DECA":
            poems_title_obj = self.tree.xpath('//text/body/div1[div2[1]]')
            poems_titles = [f'Introduzione {i.attrib["n"]}' for i in poems_title_obj]
            poems_number = len(poems_titles)
            return poems_titles, poems_number

        if path_title == "DECA_INT":
            poems_title_obj = self.tree.xpath(
                '//text/body/div1/div2[position() > 1 and not(contains(@n, "Conclusione"))]/head')
            poems_titles = [''.join(poems_title_obj[i].itertext()).translate(
                {ord(x): '' for x in self.list_avoid}) + f" - Giornata {i // 10 + 1}" for
                            i in range(len(poems_title_obj))]

            poems_number = len(poems_titles)
            return poems_titles, poems_number

        if path_title == "DECA_CONC":
            poems_title_obj = self.tree.xpath('//text/body/div1/div2[position() > 1 and (contains(@n, "Conclusione"))]')
            poems_titles = [f'{i.attrib["n"]} -  Giornata {num + 1}' for num, i in enumerate(poems_title_obj)]

            poems_number = len(poems_titles)
            return poems_titles, poems_number

        poems_title_obj = self.tree.xpath(path_title)  # getting all nodes at the head tag
        ############## getting titles #############################
        # itertext explanation https://stackoverflow.com/questions/19369901/python-element-tree-extract-text-from-element-stripping-tags
        if path_title == "DIVINA":
            poems_title_obj = self.tree.xpath('(//text/body/div1/div2/@id)')
            poems_titles = [''.join(poems_title_obj[i]).translate({ord(x): ' ' for x in self.list_avoid}) for
                            i in range(len(poems_title_obj))]
            return poems_titles, len(poems_titles)
        if path_title == "TAX":
            poems_title_obj = self.tree.xpath('(//text/body/div1[position() != 13 and  position() != 14]/@n)')
            poems_titles = [''.join(poems_title_obj[i]).translate({ord(x): ' ' for x in self.list_avoid}) for
                            i in range(len(poems_title_obj))]
            return poems_titles, len(poems_titles)
        if path_title == "TAX2":
            poems_title_obj = self.tree.xpath('(//text/body/div1[position() = 13 or  position() = 14]/@n)')
            poems_titles = [''.join(poems_title_obj[i]).translate({ord(x): ' ' for x in self.list_avoid}) for
                            i in range(len(poems_title_obj))]
            return poems_titles, len(poems_titles)

        if path_title == "VITA_NOVA":
            poems_title_obj = self.tree.xpath('//text/body/div1/lg')  # correct number of sonetti
            poems_titles = [f'{i}' for i in range(len(poems_title_obj))]
            return poems_titles, len(poems_titles)

        if path_title == "TRIONFI":
            poems_title_obj = self.tree.xpath("//text/body/div1/head")
            poems_titles = [''.join(poems_title_obj[i].itertext()).translate({ord(x): '' for x in self.list_avoid}) for
                            i in range(len(poems_title_obj))]
            poems_titles = poems_titles[3:]
            return poems_titles, len(poems_titles)

        if path_title == "AMO_VIS":

            poems_title_obj = self.tree.xpath('//text/front/div1')  # correct number of sonetti
            poems_titles = [poems_title_obj[0].attrib["n"] for i in range(3)]
            poems_number = 3
            return poems_titles, poems_number

        else:
            poems_titles = [''.join(poems_title_obj[i].itertext()).translate({ord(x): '' for x in self.list_avoid}) for
                            i in range(len(poems_title_obj))]

        poems_number = len(poems_titles)

        return poems_titles, poems_number

    def build(self, path):
        poem_obj = self.tree.xpath(path)
        # poem_obj = self.tree.xpath('//text/body/div1')
        logger.info("Author: {}".format(self.author))
        print("Number of poems :", len(poem_obj))
        logger.info("Number of poems :{} ".format(len(poem_obj)))
        print("Number of titles :", len(self.poems_titles))
        logger.info("Number of titles : {}".format(len(self.poems_titles)))
        print(self.poems_titles)
        # logger.info('Titles : ')
        poem_list = []  # list of poems in this collection
        # getting all the poems of the collection (inside the page)
        for count, obj in enumerate(poem_obj):
            poem_list.append(Poem(title=self.poems_titles[count],
                                  author=self.author,
                                  # poem_type=self.poem_type[count],
                                  collection=self.collection_title,
                                  poem_obj=obj,
                                  gl_id=self._global_id,
                                  df=self.df,
                                  df_poem=self.df_poem,
                                  family=self.family,
                                  type=self.type
                                  ))
        return poem_list

    def __call__(self):
        for i in self.poem_list:
            i.dataset_builder()


class Collection_Prose(Collection):
    def __init__(self, link, gl_id: Counter, df: Dataframe_Wrap, df_poem: Dataframe_Wrap, path_poem, path_title=None,
                 family="others", type="prose"):
        super(Collection_Prose, self).__init__(link, gl_id, df, df_poem, path_poem, path_title, family, type)

    def build(self, path):
        poem_obj = self.tree.xpath(path)
        # poem_obj = self.tree.xpath('//text/body/div1')
        logger.info("Author: {}".format(self.author))
        print("Number of proses :", len(poem_obj))
        logger.info("Number of proses :{} ".format(len(poem_obj)))
        print("Number of titles :", len(self.poems_titles))
        logger.info("Number of titles : {}".format(len(self.poems_titles)))
        print(self.poems_titles)
        # logger.info('Titles : ')
        poem_list = []  # list of poems in this collection
        # getting all the poems of the collection (inside the page)
        for count, obj in enumerate(poem_obj):
            poem_list.append(Prose(title=self.poems_titles[count],
                                   author=self.author,
                                   # poem_type=self.poem_type[count],
                                   collection=self.collection_title,
                                   poem_obj=obj,
                                   gl_id=self._global_id,
                                   df=self.df,
                                   df_poem=self.df_poem,
                                   family=self.family,
                                   type=self.type
                                   ))
        return poem_list


class Collection_200:
    def __init__(self, id, df, df_poem):
        self.global_id = id
        self.df = df

        self.df_poem = df_poem
        self.family = "'200"
        getter = requests.get("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit000709")
        # extract xml tree from page
        self.tree = etree.fromstring(getter.content)
        # get the author  and title

        self.sections = self.tree.xpath("//text/body/div1")  # list of different big contents

        self.sections_titles = [self.tree.xpath("//text/body/div1/head")[i].text for i in range(len(self.sections))]

        print(self.sections_titles)
        self.poem_list = []

    def build(self):
        for count, obj in enumerate(
                self.sections):  # obj is one big section ['TESTI ARCAICI', 'SCUOLA SICILIANA', 'POESIA CORTESE TOSCANA E SETTENTRIONALE',]
            for item in obj:  # item is a <div2> [RITMO LAURENZIANO, ..]
                for elem in item:  # check every tag in the subtree
                    if elem.tag == "head":
                        author = string.capwords(elem.text.replace(", I", '').replace(", II", '').replace("(?)", '').replace('”AMICO DI DANTE”', "Amico di Dante"))  # capitalize name
                        family = string.capwords(self.sections_titles[count])
                        if ":" in author:
                            collection = author.split(":")[1][1:]
                            author = author.split(":")[0]
                        else:
                            collection = family
                        logger.info("Author: {}".format(author))

                    elif elem.tag == "lg":  # if directly here, get the poem
                        self.poem_list.append(Poem(title=self.sections_titles[count],
                                                   author=author,
                                                   # poem_type=self.poem_type[count],
                                                   collection=collection,
                                                   family=family,
                                                   poem_obj=item,
                                                   gl_id=self.global_id,
                                                   df=self.df,
                                                   df_poem=self.df_poem
                                                   ))
                    elif elem.tag == "div3":
                        for inner in elem:
                            # avoid already processed authors
                            if "Guinizzelli" in author or "Cavalcanti" in author or "Cino" in author:
                                continue
                            if inner.tag == "head":
                                title = string.capwords(inner.text)
                            elif inner.tag == "lg":
                                self.poem_list.append(Poem(title=title,
                                                           author=author,
                                                           # poem_type=self.poem_type[count],
                                                           collection=collection,
                                                           family=family,
                                                           poem_obj=elem,
                                                           gl_id=self.global_id,
                                                           df=self.df,
                                                           df_poem=self.df_poem
                                                           ))
                            elif inner.tag == "div4":
                                for inner_last in inner:
                                    # avoid already processed authors
                                    if "Guinizzelli" in author or "Cavalcanti" in author or "Cino" in author:
                                        continue
                                    if inner_last.tag == "head":
                                        title_last = title + string.capwords(inner_last.text)
                                    elif inner_last.tag == "lg":
                                        self.poem_list.append(Poem(title=title_last,
                                                                   author=author,
                                                                   # poem_type=self.poem_type[count],
                                                                   collection=collection,
                                                                   family=family,
                                                                   poem_obj=inner,
                                                                   gl_id=self.global_id,
                                                                   df=self.df,
                                                                   df_poem=self.df_poem
                                                                   ))

    def create_db(self):
        for i in self.poem_list:
            i.dataset_builder()


# global unique identifier for all poems

global_id = Counter()

############ DATAFRAME DEFINITION ###################


columns = ['id_poem', "prev_stanza", "stanza", "author", "metrica",
           "rima"]  # , metrica (endecasillabo|ottava...), tipo_di_rima]
df = Dataframe_Wrap(columns=columns)

cols_2 = ["id_poem", "author", "title", "collection", "family", "type", "text"]
df_poem = Dataframe_Wrap(columns=cols_2)

######################### LINKS ####################




links_dolce = [
    "http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit001542",  # GUIDO GUINIZZELLI
    "http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit001585",  # GUIDO CAVALCANTI
    "http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit001110",  # CINO DA PISTOIA
    ##################
]

links_tasso = [
    "http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit000126"

]

et = Page(url="http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit000256", title=None,
          body='//text/body/div1')
var = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit000540",
           '(//text/body/div1/head)', '//text/body/div1', )
rime_amore = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit000682",
                  '(//text/body/div1/head)', '//text/body/div1', )
lacrime = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit000528",
               '(//text/body/div1/head)', '//text/body/div1', )
lacr_vergine = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit001522", "GONZAGA",
                    '//text/body/div1')
gieru = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit000169",
             '(//text/body/div1/head[1])', '//text/body/div1', )
gieru_conq = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit000260",
                  '(//text/body/div1/head[1])', '//text/body/div1', )
gonzaga = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit000126", "GONZAGA",
               '//text/body/div1', )
prologhi = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit000051",
                '(//text/body/div1/head)', '//text/body/div1', )

rime_lu = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit001217", "LUCRE",
               '//text/body/div1', )
sanbe = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit001226", "GONZAGA",
             '//text/body/div1', )
monte_oli = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit000829",
                 '(//text/body/div1/head)', '//text/body/div1', )
rinaldo = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit000970",
               '(//text/body/div1/head)', '//text/body/div1', )
floridante = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit001101",
                  '(//text/body/div1/head)', '//text/body/div1', )

geru_lib = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit001501",
                '(//text/body/div1/head)', '//text/body/div1', )
mondo_creato = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit001455",
                    '(//text/body/div1/head)', '//text/body/div1', )
pri_fer = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit001315",
               '(//text/body/div1/head)[position()>1]',
               '//text/body/div1[position()>1]', )  # prima poesia vuota (prosa)

chigiano = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit000682",
                '(//text/body/div1/head)', '//text/body/div1', )  # no strofa precedente


rime_occasione = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit000099",
                      '(//text/body/div1[2]/div2/div3[not(.//div4)]/head)',
                      '//text/body/div1[2]/div2/div3[not(.//div4)]/lg', )
rime_occasione_div4 = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit000099",
                           '(//text/body/div1[2]/div2/div3/div4[lg[not(contains(@type, "non-definito") or contains(@type, "madrigale") or contains(@type, "ottava"))]]/head)',
                           '//text/body/div1[2]/div2/div3/div4/lg[not(contains(@type, "non-definito") or contains(@type, "madrigale") or contains(@type, "ottava"))]', )  # alcuni vuoti
estra = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit001468",
             '(//text/body/div1/head)', '//text/body/div1', )  # alcune con puntini
pentito = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit001390", "TAX",
               '//text/body/div1[position() != 13 and  position() != 14]/lg', )  # alcune vuote (un lg in meno)
pentito2 = Page("http://admin.bibliotecaitaliana.netseven.it/wp-json/muruca-core/v1/xml/bibit001390", "TAX2",
                '//text/body/div1[position() = 13 or  position() = 14]', )  # alcune vuote (un lg in meno)

list_tasso = [et, var, rime_amore, lacrime, lacr_vergine, gieru, gieru_conq, gonzaga, prologhi, rime_lu, sanbe,
              monte_oli, rinaldo, floridante, geru_lib, mondo_creato, pri_fer, chigiano, rime_occasione,
              rime_occasione_div4, estra, pentito, pentito2]

###################################### ARIOSTO ##############################################

cinque_canti = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000026",
                    '(//text/body/div1/head)', '//text/body/div1', )
orlando_furioso = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit001301",
                       '(//text/body/div1/head)', '//text/body/div1', )
rime_canzoni = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000024",
                    '(//text/body/div1[1]/div2/head)', '//text/body/div1[1]/div2/lg', )
rime_sonetti = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000024",
                    '(//text/body/div1[2]/div2/head)', '//text/body/div1[2]/div2/lg', )
rime_madrigali = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000024",
                      '(//text/body/div1[3]/div2/head)', '//text/body/div1[3]/div2', )
rime_capitoli = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000024",
                     '(//text/body/div1[4]/div2/head)', '//text/body/div1[4]/div2/lg', )
satire = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000025", '(//text/body/div1/head)',
              '//text/body/div1', )

list_ariosto = [cinque_canti, orlando_furioso, rime_canzoni, rime_sonetti, rime_madrigali, rime_capitoli, satire]

print("Downloading...\n\n\n")

for i in links_dolce:
    col = Collection(link=i,
                     gl_id=global_id, df=df, df_poem=df_poem, path_poem='//text/body/div1/lg', family="Dolce Stil Novo",
                     type="poetry")
    col()

for i in list_tasso:
    col = Collection(link=i.url,
                     gl_id=global_id, df=df, df_poem=df_poem, path_poem=i.body, path_title=i.title, family="Tasso",
                     type="poetry")
    col()

for i in list_ariosto:
    col = Collection(link=i.url,
                     gl_id=global_id, df=df, df_poem=df_poem, path_poem=i.body, path_title=i.title, family="Ariosto",
                     type="poetry")
    col()

####dolce others

c200 = Collection_200(global_id, df, df_poem)
c200.build()
c200.create_db()

commedia = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000019",
                "DIVINA", '//text/body/div1/div2', family="Dolce Stil Novo", )

detto_amore = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit001448",
                   '(//titleStmt/title)', '//text/body/div1', family="Dolce Stil Novo")

il_fiore = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000213",
                '(//text/body/div1/head[1])', '//text/body/div1/lg', family="Dolce Stil Novo")

rime = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000691",
            'LUCRE', '//text/body/div1', family="Dolce Stil Novo")

conv = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit001673",
            '(//text/body/div1/div2/head)', '//text/body/div1/div2/lg', family="Dolce Stil Novo")

vita_nova = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000520",
                 'VITA_NOVA', '//text/body/div1/lg', family="Dolce Stil Novo")

#
list_dante = [commedia, detto_amore, il_fiore, rime, conv, vita_nova]

for i in list_dante:
    col = Collection(link=i.url,
                     gl_id=global_id, df=df, df_poem=df_poem, path_poem=i.body, path_title=i.title, family=i.family,
                     type="poetry")
    col()

#### PETRARCA


canzoniere = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000760",
                  '//text/body/div1/div2/head[../lg[not(contains(@type, "non-definito"))]]',
                  # select head where the lg AT THE SAME LEVEL do not contain
                  '//text/body/div1/div2/lg[not(contains(@type, "non-definito"))]', family="Petrarca", )

canzoniere2 = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000760",
                   '//text/body/div1/div2/head[../lg[(contains(@type, "non-definito"))]]',
                   # select head where the lg AT THE SAME LEVEL do not contain
                   '//text/body/div1/div2[lg[(contains(@type, "non-definito"))]]', family="Petrarca", )

frammenti = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000756",
                 "//text/body/div1/div2/head", "//text/body/div1/div2", family="Petrarca", )

rime_disperse_sonetti = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000509",
                             '//text/body/div1/div2/head[../lg[@type="sonetto"]]',
                             '//text/body/div1/div2/lg[@type="sonetto"]', family="Petrarca", )  #

rime_disperse_div3 = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000509",
                          '//text/body/div1/div2/div3/head',
                          '//text/body/div1/div2/div3', family="Petrarca", )  #

rime_disperse_other = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000509",
                           '//text/body/div1/div2/head[../lg[not(contains(@type, "sonetto"))]]',
                           '//text/body/div1/div2[lg[not(contains(@type, "sonetto"))]]', family="Petrarca", )  #

testi_del_vat = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit001257",
                     "//text/body/div1/head", "//text/body/div1", family="Petrarca", )

trionfi = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit001318",
               "//text/body/div1/div2/head", "//text/body/div1/div2/lg", family="Petrarca", )

trionfi2 = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit001318",
                "TRIONFI", "//text/body/div1/lg", family="Petrarca", )

list_petrarca = [canzoniere, canzoniere2, frammenti, rime_disperse_sonetti, rime_disperse_div3, rime_disperse_other,
                 testi_del_vat,
                 trionfi, trionfi2, ]
#
for i in list_petrarca:
    col = Collection(link=i.url,
                     gl_id=global_id, df=df, df_poem=df_poem, path_poem=i.body, path_title=i.title, family=i.family,
                     type="poetry")
    col()

## BOCCACCIO

amo_vis_intro = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000045",
                     "AMO_VIS", "//text/front/div1/lg", family="Boccaccio", )
amo_vis = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000045",
               "//text/body/div1/div2/head", "//text/body/div1/div2", family="Boccaccio", )
caccia_diana = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000044",
                    "//text/body/div1/head", "//text/body/div1", family="Boccaccio", )
canzoni_decameron = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit001339",
                         "//text/body/div1/head", "//text/body/div1/lg", family="Boccaccio", )

canzoni_comedia_ninfe = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000738",
                             '//text/body/div1/head[../lg[contains(@type, "terzina")]]',
                             '//text/body/div1[lg[contains(@type, "terzina")]]', family="Boccaccio", )

ninfale = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000047",
               '//titleStmt/title',
               '//text/body/div1', family="Boccaccio", )

rime = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000040",
            '//body/div1/div2/head[../lg[(contains(@type, "sonetto") or contains(@type, "canzone") or contains(@type, "ballata") or contains(@type, "sestina"))and not(.//gap[contains(@resp, "ed")]) ]][1]',
            '//body/div1/div2/lg[(contains(@type, "sonetto") or contains(@type, "canzone") or contains(@type, "ballata") or contains(@type, "sestina")) and not(.//gap[contains(@resp, "ed")])]',
            family="Boccaccio", )

rime2 = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000040",
             '//body/div1/div2/head[../lg[contains(@type, "terzina")] and not(.//gap[contains(@resp, "ed")])]',
             '//body/div1/div2[lg[contains(@type, "terzina")]  and not(.//gap[contains(@resp, "ed")])]',
             family="Boccaccio", )

teseida = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000041",
               '//body/div1[position() > 1]/head',
               '//body/div1[position() > 1]', family="Boccaccio", )

filostrato = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000046",
                  '//text/body/div1/head',
                  '//text/body/div1', family="Boccaccio")  # alcune vuote (un lg in meno)

list_bocaccio = [amo_vis_intro, amo_vis, caccia_diana, canzoni_decameron, canzoni_comedia_ninfe, ninfale, rime, rime2,
                 teseida, filostrato]

for i in list_bocaccio:
    col = Collection(link=i.url,
                     gl_id=global_id, df=df, df_poem=df_poem, path_poem=i.body, path_title=i.title, family=i.family,
                     type="poetry")
    col()


###################### PROSE ##########################


allegoria_gerus = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit001538",
                       "//text/body/div1/head",
                       '//text/body/div1', )

giudizio_gerus = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit001067",
                      "//text/body/div1/head",
                      '//text/body/div1', )  # alcune vuote (un lg in meno)

differenze_poe = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000580",
                      "//text/body/div1/head",
                      '//text/body/div1', )  # alcune vuote (un lg in meno)
discorsi = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000954",
                "//text/body/div1/head",
                '//text/body/div1', )  # alcune vuote (un lg in meno)

discorsi_poema_eroico = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000856",
                             "//text/body/div1/head",
                             '//text/body/div1', )  # alcune vuote (un lg in meno)

discorsi_arte_poetica = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000577",
                             "//text/body/div1/head",
                             '//text/body/div1', )  # alcune vuote (un lg in meno)

discorso_sedizione = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000641",
                          "//text/body/div1/argument/p",
                          '//text/body/div1', )  # alcune vuote (un lg in meno)

discorso_parere = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000684",
                       "//text/body/div1/head",
                       '//text/body/div1', )  # alcune vuote (un lg in meno)

messaggiero = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000679",
                   "//text/body/div1/head",
                   '//text/body/div1', )  # alcune vuote (un lg in meno)

padre_famiglia = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit001467",
                      "GONZAGA",
                      '//text/body/div1', )  # alcune vuote (un lg in meno)

secretario = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit001449",
                  "//text/body/div1[position() > 1]/head",
                  '//text/body/div1[position() > 1]', )  # alcune vuote (un lg in meno)

molza = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000749",
             "GONZAGA",
             '//text/body/div1', )  # alcune vuote (un lg in meno)

lettere = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit001167",
               "//text/body/div1/div2[not(.//div3)]/head",
               '//text/body/div1/div2[not(.//div3)]', )  # alcune vuote (un lg in meno)

lettere_poetiche = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000741",
                        "LUCRE",
                        '//text/body/div1', )  # alcune vuote (un lg in meno)

lezione_casa = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000992",
                    "LUCRE",
                    '//text/body/div1', )  # alcune vuote (un lg in meno)

orazioni = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000077",
                "LUCRE",
                '//text/body/div1', )  # alcune vuote (un lg in meno)

crusca = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000863",
              "//text/body/div1/head",
              '//text/body/div1', )  # alcune vuote (un lg in meno)
plutarco = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000391",
                "//text/body/div1/head",
                '//text/body/div1', )  # alcune vuote (un lg in meno)
dignita = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000788",
               "//text/body/div1/head",
               '//text/body/div1', )  # alcune vuote (un lg in meno)

list_tasso = [allegoria_gerus, giudizio_gerus, differenze_poe, discorsi, discorsi_poema_eroico, discorsi_arte_poetica,
              discorso_sedizione, discorso_parere, messaggiero, padre_famiglia, secretario, molza, lettere,
              lettere_poetiche, lezione_casa, orazioni, crusca, plutarco, dignita]

for i in list_tasso:
    col = Collection_Prose(link=i.url,
                           gl_id=global_id, df=df, df_poem=df_poem, path_poem=i.body, path_title=i.title,
                           family="Tasso",
                           type="prose")
    col()

convivio = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit001673",
                "//text/body/div1/div2[not(.//lg)]/head",
                '//text/body/div1/div2[not(.//lg)]', )  # alcune vuote (un lg in meno)

vita_nuova = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000520",
                  "//text/body/div1/head",
                  '//text/body/div1', )  # alcune vuote (un lg in meno)

list_dante = [convivio, vita_nuova]

for i in list_dante:
    col = Collection_Prose(link=i.url,
                           gl_id=global_id, df=df, df_poem=df_poem, path_poem=i.body, path_title=i.title,
                           family="Dolce Stil Novo",
                           type="prose")
    col()

# ARIOSTO


lettere = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit001276",
               "//text/body/div1/head",
               '//text/body/div1', )  # alcune vuote (un lg in meno)
guardaroba = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit001106",
                  "LUCRE2",
                  '//text/body/div1/div2', )  # alcune vuote (un lg in meno)

list_ariosto = [lettere, guardaroba]

#
for i in list_ariosto:
    col = Collection_Prose(link=i.url,
                           gl_id=global_id, df=df, df_poem=df_poem, path_poem=i.body, path_title=i.title,
                           family="Ariosto",
                           type="prose")
    col()

#### PETRARCA

epistole_senili = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000395",
                       "//text/body/div1/div2/head[1]",
                       '//text/body/div1/div2', )  # alcune vuote (un lg in meno)

list_petrarca = [epistole_senili]

#
for i in list_petrarca:
    col = Collection_Prose(link=i.url,
                           gl_id=global_id, df=df, df_poem=df_poem, path_poem=i.body, path_title=i.title,
                           family="Petrarca",
                           type="prose")
    col()

## BOCCACCIO

comedia_ninfe = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000738",
                     "//text/body/div1[not(.//lg)]/head",
                     '//text/body/div1[not(.//lg)]', )  # alcune vuote (un lg in meno)
consolatoria_pino = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit001053",
                         "//text/body/div1/head",
                         '//text/body/div1', )  # alcune vuote (un lg in meno)
corbaccio = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000039",
                 "//titleStmt/title",
                 '//text/body/div1', )  # alcune vuote (un lg in meno)

elegia_fiammetta = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000043",
                        "//text/body/div1/div2/head",
                        '//text/body/div1/div2', )  # alcune vuote (un lg in meno)

epistole_lettere = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000919",
                        '//text/body/div1[contains(@lang, "it")]/head',
                        '//text/body/div1[contains(@lang, "it")]', )  # alcune vuote (un lg in meno)
epistole_lettere2 = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000919",
                         '//text/body/div1[div2[contains(@n, "2")]]/head',
                         '//text/body/div1[div2[contains(@n, "2")]]/div2[contains(@lang, "it")]', )  # alcune vuote (un lg in meno)

epistole_lettere3 = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000919",
                         '//text/body/div1/div2[contains(@lang, "it") and not(contains(@n, "2")) ]/head',
                         '//text/body/div1/div2[contains(@lang, "it") and not(contains(@n, "2")) ]', )  # alcune vuote (un lg in meno)

filoloco = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000042",
                '//text/body/div1/div2/head',
                '//text/body/div1/div2', )  # alcune vuote (un lg in meno)
filostrato_proemio = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000046",
                          '//text/front/div1/head',
                          '//text/front/div1[.//head]', )  # alcune vuote (un lg in meno)

teseida = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000041",
               '//body/div1[position() = 1]/head',
               '//body/div1[position() = 1]', family="Boccaccio", )

trattatello = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000101",
                   '//body/div1/head',
                   '//body/div1', family="Boccaccio", )

decameron_proemio = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000267",
                         "//text/front/div1[position() = 1]/head",
                         '//text/front/div1[position() = 1]', )  # alcune vuote (un lg in meno)

decameron_intros = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000267",
                        'DECA',
                        '//body/div1/div2[1]', family="Boccaccio", )

decameron = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000267",
                 'DECA_INT',
                 '//text/body/div1/div2[position() > 1 and not(contains(@n, "Conclusione"))]', )  # alcune vuote (un lg in meno)  # TODO

decameron_conclude = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000267",
                          'DECA_CONC',
                          '//text/body/div1/div2[position() > 1 and (contains(@n, "Conclusione"))]', )  # alcune vuote (un lg in meno)  # TODO

decameron_bocc_conclude = Page("http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/xml/bibit000267",
                               '//text/body/div1[contains(@type, "parte")]/head',
                               '//text/body/div1[contains(@type, "parte")]', )  # alcune vuote (un lg in meno)  # TODO

list_boccaccio = [comedia_ninfe, consolatoria_pino, corbaccio, elegia_fiammetta, epistole_lettere, epistole_lettere2,
                  epistole_lettere3, filoloco, filostrato_proemio, teseida, trattatello, decameron_proemio,
                  decameron_intros, decameron, decameron_conclude, decameron_bocc_conclude]

for i in list_boccaccio:
    col = Collection_Prose(link=i.url,
                           gl_id=global_id, df=df, df_poem=df_poem, path_poem=i.body, path_title=i.title,
                           family="Boccaccio",
                           type="prose")
    col()


print("Download completed!")

df_poem = df_poem.df.replace({'family' : { 'Poesia Didattica Del Nord' : "Northern Didactic poetry",
                                             'Dolce Stil Novo': "Stilnovisti",
                              'Poesia Cortese Toscana E Settentrionale' : "Northern/Tuscan Courtly poetry",
                              "Poesia Didattica Dell'italia Centrale" : "Central Italy Didactic poetry",
                              "Poesia Popolare E Giullaresca": "Folk and Giullaresca poetry",
                              "Poesia “realistica” Toscana":"Realistic Tuscan poetry",
                              "Scuola Siciliana":"Sicilian school",
                             "Testi Arcaici":"Archaic text",
                             "Vicini Degli Stilnovisti": "Similar to Stilnovisti"}})
# df.to_csv("stanzas.csv", sep=";")
df_poem.to_csv("vulgaris.csv", sep=";")

print("Dataset downloaded and formatted!")
