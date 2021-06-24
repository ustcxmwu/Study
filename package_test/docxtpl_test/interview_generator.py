import pprint as pp
from random import sample
from easydict import EasyDict as edict

import yaml
from docxtpl import DocxTemplate
from jinja2 import Environment


def is_list(value):
    return isinstance(value, list)


if __name__ == '__main__':
    jinja_env = Environment()
    jinja_env.filters["is_list"] = is_list
    tpl = DocxTemplate('interview_report_tpl.docx')
    context = edict({
        "type": "社招",
        "name": "高凯戈",
        "university": "哈工大-纽约大学布法罗分校",
        # "major": "物理学类",
        "projects_file": "gaokaige.yml"
    })

    with open(context["projects_file"], mode='r', encoding="utf-8") as f:
        projects = yaml.safe_load(f)
    # pp.pprint(projects)
    context["projects"] = projects
    with open("ai.yml", mode='r', encoding="utf-8") as f:
        ai_questions = yaml.safe_load(f)
    context["ai_questions"] = sample(ai_questions, min(10, len(ai_questions)))
    with open("program_questions.yml", mode='r', encoding='utf-8') as f:
        program_questions = yaml.safe_load(f)
    context["program_questions"] = sample(program_questions, min(10, len(program_questions)))
    pp.pprint(context)
    tpl.render(context, jinja_env=jinja_env)
    tpl.save('./out/{}_{}_面试情况.docx'.format(context.type, context.name))
