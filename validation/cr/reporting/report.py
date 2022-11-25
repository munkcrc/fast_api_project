from pathlib import Path
from typing import Dict, List
from uuid import uuid4
import yaml

class Section(object):
    def __init__(self, title:str, content:str):
        self.title = title
        self.content = content

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "content": self.content
        }

    @classmethod
    def from_dict(cls, dict):
        title = dict["title"] if dict["title"] else "Section - " + uuid4().hex[:8]
        content = dict["content"]
        return cls(title, content)

class Chapter(object):
    __chpter_count = 0

    def __init__(self, title:str, sections:List[Section]):
        self.title = title
        self.sections = sections

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "sections": [section.to_dict() for section in self.sections]
        }

    @classmethod
    def from_dict(cls, dict):
        title = dict["title"] if dict["title"] else "Chapter - " + uuid4().hex[:8]
        sections =  [Section.from_dict(section) for section in dict['sections']]
        return cls(title, sections)

class Report(object):

    def __init__(self, title:str, chapters:List[Chapter], context:Dict=None):
        self.title = title
        self.chapters = chapters

        if not context:
            context = {}
        if not '_TITLE' in context:
            context['_TITLE'] = self.title

        self.context = context

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "chapters": [chapter.to_dict() for chapter in self.chapters],
            "context": self.context
        }       

    def to_yaml(self, stream=None):
        return yaml.dump(self.to_dict(), stream)
    
    @classmethod
    def from_dict(cls, dict):
        title = dict['title'] if dict['title'] else 'Report'
        chapters = [Chapter.from_dict(chapter) for chapter in dict['chapters']]
        context = dict['context']
        return cls(title, chapters, context)

    @classmethod
    def from_yaml(cls, stream):
        # If the stream is not a stream but a path we open the stream for convenience
        if isinstance(stream, str) or isinstance(stream, Path):
            with open(stream, 'rb') as file:
                dict = yaml.safe_load(file)
        else:
            dict = yaml.safe_load(stream)
        return cls.from_dict(dict)