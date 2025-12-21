# modules/utils.py
import re
import jieba

def extract_price(text: str):
    """提取预算区间"""
    match = re.search(r'(\d+)\s*[-到至~]\s*(\d+)', text)
    if match:
        return tuple(map(int, match.groups()))
    match = re.search(r'(\d+)', text)
    if match:
        return (0, int(match.group(1)))
    return None

def extract_keywords(text: str):
    """分词 + 去重"""
    return set(jieba.lcut(text))
