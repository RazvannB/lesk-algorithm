import os
import re
import xml.etree.ElementTree as Et


def get_instance(instance_el):
    """
    :param instance_el: instance element from xml
    :return: dict version of instance
    """
    instance = instance_el[0].attrib
    instance['instanceid'] = instance_el.attrib.get('id', '')
    context_el = instance_el[1]

    # get raw context
    raw_context = Et.tostring(context_el, 'utf-8', method="xml", short_empty_elements=True).decode('utf-8')
    re_tword = re.compile(r'(?:<head>)(.*)(?:</head>)')
    re_sentence = re.compile(r'(?:<s>)(.*)(?:</s>)')
    re_tokens = re.compile(r'\w+')
    # get target word
    target_word = re_tword.findall(raw_context)
    # remove from context head tags and let only target word
    context_nohead = re_tword.sub(target_word[0], raw_context)
    # remove context tags
    raw_text = ' '.join(re_sentence.findall(context_nohead))
    text = ' '.join(re_tokens.findall(raw_text))

    instance['text'] = text
    instance['target_word'] = target_word[0]

    print(instance)


def main():
    """Test evaluation methods"""
    # define file paths
    file_name = 'hard-a.train.xml'
    file_path = os.path.join('data_eval', file_name)

    # read xml
    tree = Et.parse(file_path)
    root = tree.getroot()
    lexelt = root[0]
    instance_el = lexelt[0]

    get_instance(instance_el)


if __name__ == '__main__':
    main()
