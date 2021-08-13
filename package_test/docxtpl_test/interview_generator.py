import pprint as pp
from random import sample
from easydict import EasyDict as edict

import yaml
from docxtpl import DocxTemplate
from jinja2 import Environment


def is_list(value):
    return isinstance(value, list)


def get_tpl(interview_type: str):
    if interview_type == "实习生":
        return DocxTemplate("./tpl/intern_interview_report_tpl.docx")
    elif interview_type == "校招":
        return DocxTemplate("./tpl/campas_interview_report_tpl.docx")
    elif interview_type == "社招":
        return DocxTemplate("./tpl/social_interview_report_tpl.docx")
    else:
        raise ValueError("undefined interview_type: {}".format(interview_type))


if __name__ == '__main__':
    context = edict({
        "type": "校招",
        "name": "严茹丹",
        "university": "华中科技大学",
        # "major": "物理学类",
        "projects_file": "yangrudan.yml"
    })
    tpl = get_tpl(context.type)

    with open(context["projects_file"], mode='r', encoding="utf-8") as f:
        projects = yaml.safe_load(f)
    # pp.pprint(projects)
    context["projects"] = projects
    with open("questions/ai.yml", mode='r', encoding="utf-8") as f:
        ai_questions = yaml.safe_load(f)
    context["ai_questions"] = sample(ai_questions, min(10, len(ai_questions)))
    with open("questions/program_questions.yml", mode='r', encoding='utf-8') as f:
        program_questions = yaml.safe_load(f)
    context["program_questions"] = sample(program_questions, min(10, len(program_questions)))
    pp.pprint(context)

    jinja_env = Environment()
    jinja_env.filters["is_list"] = is_list
    tpl.render(context, jinja_env=jinja_env)
    tpl.save('./out/{}_{}_面试情况.docx'.format(context.type, context.name))
