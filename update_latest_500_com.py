#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
从500.com获取最新一期双色球数据并添加到CSV文件顶部
"""
import requests
import os
import re
import time
from bs4 import BeautifulSoup
from datetime import datetime

class SSQLatestFetcher:
    def __init__(self):
        self.base_url = "http://kaijiang.500.com/ssq.shtml"
        self.data_file = "/home/user/kkde_SSQ/双色球/ssq_history.csv"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Referer': 'http://kaijiang.500.com/ssq.shtml',
        }
        
    def convert_period_format(self, period_str):
        """
        转换期号格式
        500.com返回的期号可能是5位数字（如25127），需要转换为7位格式（如2025127）
        """
        # 移除所有非数字字符
        period = re.sub(r'\D', '', str(period_str))
        
        if len(period) == 5:
            # 5位期号：前2位是年份后两位，后3位是期数
            year_suffix = period[:2]
            period_num = period[2:]
            
            # 判断年份：00-25可能是2000-2025，25-99可能是2025-2099
            if int(year_suffix) <= 25:
                year = 2000 + int(year_suffix)
            else:
                year = 1900 + int(year_suffix)
            
            return f"{year}{period_num}"
        elif len(period) == 7:
            # 已经是7位格式，直接返回
            return period
        else:
            # 其他格式，尝试修复
            print(f"  警告: 期号格式异常: {period_str} (长度: {len(period)})")
            return period
    
    def get_latest_period_url(self):
        """
        获取最新一期的URL
        """
        try:
            response = requests.get(self.base_url, headers=self.headers, timeout=15)
            response.encoding = 'gb2312'
            
            if response.status_code != 200:
                print(f"请求失败，状态码: {response.status_code}")
                return None, None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            pageList = soup.select("div.iSelectList a")
            
            if len(pageList) == 0:
                print("未找到期号链接")
                return None, None
            
            # 第一个链接就是最新一期
            latest_link = pageList[0]
            url = latest_link.get('href', '')
            page_text = latest_link.string if latest_link.string else latest_link.get_text().strip()
            
            if not url.startswith('http'):
                url = "http://kaijiang.500.com" + url
            
            return url, page_text
            
        except Exception as e:
            print(f"获取最新期号失败: {e}")
            return None, None
    
    def download_period_data(self, url, page_str):
        """
        从详情页面下载数据
        """
        try:
            session = requests.Session()
            session.headers.update(self.headers)
            
            response = session.get(url, timeout=15)
            response.encoding = 'gb2312'
            
            if response.status_code != 200:
                print(f"请求失败，状态码: {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找号码元素
            ball_list = soup.select('div.ball_box01 ul li')
            
            if len(ball_list) < 7:
                print(f"  警告: 号码元素不足，期望7个，实际{len(ball_list)}个")
                return None
            
            # 提取号码
            ball = []
            for li in ball_list[:7]:
                num = li.string if li.string else li.get_text().strip()
                ball.append(num)
            
            # 转换期号格式
            period = self.convert_period_format(page_str)
            
            # 提取开奖日期
            date = self.extract_date(soup)
            
            # 验证数据
            if len(ball) == 7:
                red_balls = [int(b) for b in ball[:6]]
                blue_ball = int(ball[6])
                
                # 验证红球范围（1-33）且不重复
                if len(set(red_balls)) != 6 or any(r < 1 or r > 33 for r in red_balls):
                    print(f"  警告: 红球数据异常: {red_balls}")
                    return None
                
                # 验证蓝球范围（1-16）
                if blue_ball < 1 or blue_ball > 16:
                    print(f"  警告: 蓝球数据异常: {blue_ball}")
                    return None
                
                # 排序红球
                red_balls.sort()
                
                return {
                    '期号': period,
                    '开奖日期': date,
                    '红球1': str(red_balls[0]),
                    '红球2': str(red_balls[1]),
                    '红球3': str(red_balls[2]),
                    '红球4': str(red_balls[3]),
                    '红球5': str(red_balls[4]),
                    '红球6': str(red_balls[5]),
                    '蓝球': str(blue_ball)
                }
            
            return None
            
        except Exception as e:
            print(f"下载数据失败: {e}")
            return None
    
    def extract_date(self, soup):
        """
        从详情页面提取开奖日期
        """
        try:
            date_patterns = [
                r'(\d{4}[-年]\d{1,2}[-月]\d{1,2})',
                r'(\d{4}/\d{1,2}/\d{1,2})',
                r'开奖日期[：:]\s*(\d{4}[-年]\d{1,2}[-月]\d{1,2})',
            ]
            
            text = soup.get_text()
            for pattern in date_patterns:
                match = re.search(pattern, text)
                if match:
                    date_str = match.group(1)
                    date_str = date_str.replace('年', '-').replace('月', '-').replace('/', '-')
                    return date_str
            
            return ""
        except:
            return ""
    
    def check_period_exists(self, period):
        """
        检查期号是否已存在于CSV文件中（精确匹配第一个字段）
        """
        if not os.path.exists(self.data_file):
            return False
        
        try:
            with open(self.data_file, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    line = line.strip()
                    # 跳过空行和表头
                    if not line or line.startswith('期号'):
                        continue
                    # 精确匹配第一个字段（期号列）
                    fields = line.split(',')
                    if len(fields) > 0 and fields[0] == period:
                        return True
            return False
        except Exception as e:
            print(f"检查期号时出错: {e}")
            return False
    
    def insert_to_top(self, data):
        """
        将数据插入到CSV文件的第一行（表头之后）
        注意：调用此方法前应该先检查期号是否存在
        """
        if data is None:
            return False
        
        period = data['期号']
        
        # 双重检查：再次确认期号不存在（防止并发情况）
        if self.check_period_exists(period):
            print(f"期号 {period} 已存在于文件中，跳过添加")
            return False
        
        # 读取现有文件内容
        lines = []
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
        else:
            # 如果文件不存在，创建表头
            lines = ['期号,开奖日期,红球1,红球2,红球3,红球4,红球5,红球6,蓝球\n']
        
        # 构建新数据行
        new_line = f"{data['期号']},{data['开奖日期']},{data['红球1']},{data['红球2']},{data['红球3']},{data['红球4']},{data['红球5']},{data['红球6']},{data['蓝球']}\n"
        
        # 将新数据插入到表头之后（第二行）
        if len(lines) > 0:
            lines.insert(1, new_line)
        else:
            lines.append(new_line)
        
        # 写回文件
        try:
            with open(self.data_file, 'w', encoding='utf-8-sig') as f:
                f.writelines(lines)
            return True
        except Exception as e:
            print(f"写入文件失败: {e}")
            return False
    
    def fetch_latest(self):
        """
        获取最新一期数据并添加到文件顶部
        """
        print("正在获取最新一期双色球数据...")
        
        # 获取最新期号URL
        url, page_text = self.get_latest_period_url()
        if not url or not page_text:
            print("无法获取最新期号")
            return
        
        # 转换期号格式
        period = self.convert_period_format(page_text)
        
        # 先检查期号是否已存在
        if self.check_period_exists(period):
            print(f"期号 {period} 已存在于文件中，无需添加")
            return
        
        print(f"最新期号: {page_text} (格式化后: {period})")
        print(f"正在下载数据...")
        
        # 下载数据
        data = self.download_period_data(url, page_text)
        if not data:
            print("下载数据失败")
            return
        
        print(f"获取成功！")
        print(f"期号: {data['期号']}")
        print(f"开奖日期: {data['开奖日期']}")
        print(f"红球: {data['红球1']} {data['红球2']} {data['红球3']} {data['红球4']} {data['红球5']} {data['红球6']}")
        print(f"蓝球: {data['蓝球']}")
        
        # 再次检查（防止下载过程中数据变化）
        if self.check_period_exists(data['期号']):
            print(f"\n期号 {data['期号']} 已存在于文件中，跳过添加")
            return
        
        # 插入到文件顶部
        if self.insert_to_top(data):
            print(f"\n✓ 数据已成功添加到文件顶部: {self.data_file}")
        else:
            print(f"\n✗ 数据添加失败")


def main():
    fetcher = SSQLatestFetcher()
    fetcher.fetch_latest()


if __name__ == '__main__':
    main()

