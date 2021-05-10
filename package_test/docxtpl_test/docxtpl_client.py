from docxtpl import DocxTemplate

if __name__ == '__main__':
    tpl = DocxTemplate('tpl.docx')
    context = {
        'who': '程旭阳'
    }
    tpl.render(context)
    tpl.save('leave.docx')
