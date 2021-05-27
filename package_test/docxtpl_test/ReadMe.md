## 使用说明

### 文件说明
ai.yaml  AI面试题库
program_questions.yml 编程面试题库
projects.yml 项目相关
interview_report_tpl.docx 面试情况模板
interview_generator.py 主程序

### 使用过程
1. 复制 projects.yml 为新的文件例如 zhaohaonan.yml
2. 根据被面试人员的简历添加感兴趣的项目和问题
3. 修改主程序 interview_generator.py 18行name为面试者名称，19行project为1步骤中的yml文件，运行程序
4. 在out路径下可以看到name_面试情况.docx的文件
