import json


class OntologyProcessing(object):

    @staticmethod
    def read_ontology(filename):
        # 0. read AudioSet Ontology data
        with open(filename) as data_file:
            raw_aso = json.load(data_file)
        return raw_aso

    @staticmethod
    def form_dictionary(raw_aso):
        # 1. format data as a dictionary
        ## aso["/m/0dgw9r"] > {'restrictions': [u'abstract'], 'child_ids': [u'/m/09l8g', u'/m/01w250', u'/m/09hlz4', u'/m/0bpl036', u'/m/0160x5', u'/m/0k65p', u'/m/01jg02', u'/m/04xp5v', u'/t/dd00012'], 'name': u'Human sounds'}
        aso = {}
        for category in raw_aso:
            tmp = dict()
            tmp["name"] = category["name"]
            tmp["restrictions"] = category["restrictions"]
            tmp["child_ids"] = category["child_ids"]
            tmp["parents_ids"] = []
            aso[category["id"]] = tmp
        return aso

    @staticmethod
    def fetch_parents(aso):
        # 2. fetch higher_categories > ["/m/0dgw9r","/m/0jbk","/m/04rlf","/t/dd00098","/t/dd00041","/m/059j3w","/t/dd00123"]
        for cat in aso: # find parents
            for c in aso[cat]["child_ids"]:
                aso[c]["parents_ids"].append(cat)
        return aso

    @staticmethod
    def get_label_name_list(filename):
        raw_aso = OntologyProcessing.read_ontology(filename)
        aso = OntologyProcessing.form_dictionary(raw_aso)
        aso = OntologyProcessing.fetch_parents(aso)
        return aso

    @staticmethod
    def get_2nd_level_label_name_list(filename):
        aso = OntologyProcessing.get_label_name_list(filename)
        first_level_class = {}
        for key, value in aso.items():
            if not value['parents_ids']:
                first_level_class[key] = value

        second_level_class = {}
        for key, value in aso.items():
            try:
                if key == "/m/0395lw":  #special case, the Bell could either be Music or Sounds of things, make it sound of things
                    value['parents_ids'] = [value['parents_ids'][1]]

                if value['parents_ids'][0] in first_level_class.keys():
                    second_level_class[key] = value
            except IndexError:
                continue
        second_level_class[u'/m/04rlf'] = aso['/m/04rlf'] #special case, many data has a class - Music
        return second_level_class.keys(), aso

    @staticmethod
    def get_2nd_level_class_label_index(label_code, aso, second_level_class):
        # aso = OntologyProcessing.get_label_name_list(filename)
        # second_level_class = OntologyProcessing.get_2nd_level_label_name_list(filename)
        class_set = set()
        for idx in range(0, len(label_code)):
            buf = label_code[idx]
            while buf not in second_level_class:
                try:
                    buf = aso[buf]['parents_ids'][0]
                except IndexError:
                    break
            class_set.add(buf)

        return list(class_set)