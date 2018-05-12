import json
import os
import re
import xml.etree.ElementTree as Et


def get_instance(instance_el):
    """
    :param instance_el: instance element from xml
    :return: dict version of instance
    """

    re_tword = re.compile(r'(?:<head>)(.*)(?:</head>)')
    re_sentence = re.compile(r'(?:<s>)(.*)(?:</s>)')
    re_tokens = re.compile(r'[\w\,\.\']+')

    instance = instance_el[0].attrib
    instance['instanceid'] = instance_el.attrib.get('id', '')
    context_el = instance_el[1]

    # get raw context
    raw_context = Et.tostring(context_el, 'utf-8', method="xml", short_empty_elements=True).decode('utf-8')
    # get target word
    target_word = re_tword.findall(raw_context)
    # remove from context head tags and let only target word
    context_nohead = re_tword.sub(target_word[0], raw_context)
    # remove context tags
    raw_text = ' '.join(re_sentence.findall(context_nohead))
    text = ' '.join(re_tokens.findall(raw_text))

    instance['text'] = text
    instance['target_word'] = target_word[0]

    return instance


def main():
    """Test evaluation methods"""
    # define file paths
    # xml_file_name = 'hard-a.train.xml'
    # xml_file_name = 'interest-n.xml'
    xml_file_name = 'line-n.train.xml'
    # json_file_name = 'hard-a.train.json'
    # json_file_name = 'interest-n.json'
    json_file_name = 'line-n.train.json'

    xml_file_path = os.path.join('data_eval', xml_file_name)
    json_file_path = os.path.join('data_eval', json_file_name)

    # read xml
    tree = Et.parse(xml_file_path)
    root = tree.getroot()
    lexelt = root[0]
    print(get_instance(lexelt[0]))

    # save json version
    instances = [get_instance(instance_el) for instance_el in lexelt]
    with open(json_file_path, 'w') as file:
        json.dump(instances, file, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
