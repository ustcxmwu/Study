import pprint as pp
from random import sample
from easydict import EasyDict as edict

import yaml
from docxtpl import DocxTemplate
from jinja2 import Environment


points = {
    "实习生": 3,
    "校招": 4,
    "社招": 5
}

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


def get_ai_questions(interview_type: str):
    level = points[interview_type]
    with open("questions/ai.yml", mode='r', encoding="utf-8") as f:
        ai_questions = yaml.safe_load(f)
    questions = [ai for ai in ai_questions if ai["point"] < level]
    return sample(questions, min(10, len(questions)))


def get_program_questions(interview_type: str):
    level = points[interview_type]
    with open("questions/program_questions.yml", mode='r', encoding='utf-8') as f:
        program_questions = yaml.safe_load(f)
    questions = [q for q in program_questions if q["point"] < level]
    return sample(questions, min(10, len(questions)))


if __name__ == '__main__':
    context = edict({
        "type": "社招",
        "name": "王梦硕",
        "university": "复旦（微电子）- 复旦（微电子硕博连读）",
        # "major": "物理学类",
        "projects_file": "wangmengshuo.yml"
    })
    tpl = get_tpl(context.type)

    with open(context["projects_file"], mode='r', encoding="utf-8") as f:
        projects = yaml.safe_load(f)
    # pp.pprint(projects)
    context["projects"] = projects
    context["ai_questions"] = get_ai_questions(context.type)
    context["program_questions"] = get_program_questions(context.type)
    pp.pprint(context)

    jinja_env = Environment()
    jinja_env.filters["is_list"] = is_list
    tpl.render(context, jinja_env=jinja_env)
    tpl.save('./out/{}_{}_面试情况.docx'.format(context.type, context.name))
