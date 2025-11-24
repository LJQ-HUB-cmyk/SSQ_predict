#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
只更新最新一期的双色球开奖数据
从78500.cn或500.com获取最新期号的数据
"""
import requests
import pandas as pd
import os
import re
from bs4 import BeautifulSoup
from datetime import datetime

class SSQLatestUpdater:
    def __init__(self, data_file="ssq_history.csv"):
        self.data_file = data_file
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }
    
    def get_latest_period_from_file(self):
        """
        从CSV文件中获取最新期号
        """
        if not os.path.exists(self.data_file):
            print(f"文件 {self.data_file} 不存在")
            return None
        
        try:
            df = pd.read_csv(self.data_file, encoding='utf-8-sig')
            if len(df) == 0:
                return None
            
            # 按期号排序（降序）
            df = df.sort_values('期号', ascending=False)
            latest_period = df.iloc[0]['期号']
            return str(int(latest_period))  # 转换为字符串并去掉可能的.0
        except Exception as e:
            print(f"读取文件失败: {e}")
            return None
    
    def get_latest_from_78500(self):
        """
        从78500.cn获取最新一期数据
        """
        try:
            url = "https://m.78500.cn/kaijiang/ssq/"
            session = requests.Session()
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            }
            
            response = session.get(url, headers=headers, timeout=15)
            response.encoding = 'gb2312'
            
            if response.status_code != 200:
                print(f"  请求失败，状态码: {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找第一个 <section class="item"> 元素（最新一期）
            first_item = soup.find('section', class_='item')
            if not first_item:
                print("  未找到最新一期数据")
                return None
            
            # 提取期号
            strong_tag = first_item.find('strong')
            if not strong_tag:
                return None
            
            period_text = strong_tag.get_text().strip()
            period_match = re.search(r'((19|20)\d{5})', period_text)
            if not period_match:
                return None
            period = period_match.group(1)
            
            # 提取日期
            date = ""
            span_tag = first_item.find('span')
            if span_tag:
                date_text = span_tag.get_text().strip()
                date_match = re.search(r'(\d{4}-\d{1,2}-\d{1,2})', date_text)
                if date_match:
                    date = date_match.group(1)
            
            # 提取号码
            p_tag = first_item.find('p')
            if not p_tag:
                return None
            
            # 提取红球：6个 <i> 标签
            red_balls = []
            i_tags = p_tag.find_all('i')
            for i_tag in i_tags:
                num_text = i_tag.get_text().strip()
                try:
                    num = int(num_text)
                    if 1 <= num <= 33 and num not in red_balls:
                        red_balls.append(num)
                except ValueError:
                    continue
            
            # 提取蓝球：1个 <b> 标签
            blue_ball = None
            b_tag = p_tag.find('b')
            if b_tag:
                blue_text = b_tag.get_text().strip()
                try:
                    blue_ball = int(blue_text)
                    if not (1 <= blue_ball <= 16):
                        blue_ball = None
                except ValueError:
                    pass
            
            # 验证数据
            if len(red_balls) == 6 and blue_ball is not None:
                red_balls = sorted(red_balls)
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
            print(f"  从78500.cn获取失败: {e}")
            return None
    
    def get_latest_from_500(self):
        """
        从500.com获取最新一期数据
        """
        try:
            url = "http://kaijiang.500.com/ssq.shtml"
            response = requests.get(url, headers=self.headers, timeout=15)
            response.encoding = 'gb2312'
            
            if response.status_code != 200:
                print(f"  请求失败，状态码: {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找第一个期号链接（最新一期）
            pageList = soup.select("div.iSelectList a")
            if len(pageList) == 0:
                print("  未找到期号链接")
                return None
            
            first_link = pageList[0]
            detail_url = first_link.get('href', '')
            page_text = first_link.string if first_link.string else first_link.get_text().strip()
            
            if not detail_url or not page_text:
                return None
            
            # 获取详情页面
            if not detail_url.startswith('http'):
                detail_url = "http://kaijiang.500.com" + detail_url
            
            detail_response = requests.get(detail_url, headers=self.headers, timeout=15)
            detail_response.encoding = 'gb2312'
            detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
            
            # 查找号码
            ball_list = detail_soup.select('div.ball_box01 ul li')
            if len(ball_list) < 7:
                return None
            
            # 提取号码
            ball = []
            for li in ball_list[:7]:
                num = li.string if li.string else li.get_text().strip()
                ball.append(num)
            
            # 转换期号格式
            period = self.convert_period_format(page_text)
            
            # 提取日期
            date = self.extract_date(detail_soup)
            
            # 验证数据
            if len(ball) == 7:
                red_balls = [int(b) for b in ball[:6]]
                blue_ball = int(ball[6])
                
                # 验证
                if len(set(red_balls)) != 6 or any(r < 1 or r > 33 for r in red_balls):
                    return None
                if blue_ball < 1 or blue_ball > 16:
                    return None
                
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
            print(f"  从500.com获取失败: {e}")
            return None
    
    def convert_period_format(self, period_str):
        """转换期号格式（5位转7位）"""
        period = re.sub(r'\D', '', str(period_str))
        if len(period) == 5:
            year_suffix = period[:2]
            period_num = period[2:]
            if int(year_suffix) <= 25:
                year = 2000 + int(year_suffix)
            else:
                year = 1900 + int(year_suffix)
            return f"{year}{period_num}"
        elif len(period) == 7:
            return period
        else:
            return period
    
    def extract_date(self, soup):
        """从详情页面提取开奖日期"""
        try:
            date_patterns = [
                r'(\d{4}[-年]\d{1,2}[-月]\d{1,2})',
                r'(\d{4}/\d{1,2}/\d{1,2})',
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
    
    def append_to_csv(self, data):
        """追加数据到CSV文件"""
        if data is None:
            return False
        
        try:
            # 检查文件是否存在
            file_exists = os.path.exists(self.data_file)
            
            # 读取现有数据检查是否已存在
            if file_exists:
                df = pd.read_csv(self.data_file, encoding='utf-8-sig')
                if data['期号'] in df['期号'].values:
                    print(f"  期号 {data['期号']} 已存在于文件中，跳过")
                    return False
            
            # 追加新数据
            with open(self.data_file, 'a', encoding='utf-8-sig') as f:
                if not file_exists:
                    # 写入表头
                    f.write('期号,开奖日期,红球1,红球2,红球3,红球4,红球5,红球6,蓝球\n')
                # 写入数据
                f.write(f"{data['期号']},{data['开奖日期']},{data['红球1']},{data['红球2']},{data['红球3']},{data['红球4']},{data['红球5']},{data['红球6']},{data['蓝球']}\n")
            
            return True
            
        except Exception as e:
            print(f"写入文件失败: {e}")
            return False
    
    def update(self, source='auto'):
        """
        更新最新一期数据
        source: 'auto' (自动选择), '78500' (78500.cn), '500' (500.com)
        """
        print("=" * 60)
        print("检查并更新最新一期双色球开奖数据")
        print("=" * 60)
        
        # 获取本地最新期号
        local_latest = self.get_latest_period_from_file()
        if local_latest:
            print(f"本地最新期号: {local_latest}")
        else:
            print("本地无数据，将获取最新一期")
        
        # 获取网站最新数据
        latest_data = None
        
        if source == 'auto':
            # 先尝试78500.cn
            print("\n尝试从 78500.cn 获取最新数据...")
            latest_data = self.get_latest_from_78500()
            
            if not latest_data:
                print("\n尝试从 500.com 获取最新数据...")
                latest_data = self.get_latest_from_500()
        elif source == '78500':
            print("\n从 78500.cn 获取最新数据...")
            latest_data = self.get_latest_from_78500()
        elif source == '500':
            print("\n从 500.com 获取最新数据...")
            latest_data = self.get_latest_from_500()
        else:
            print(f"未知的数据源: {source}")
            return
        
        if not latest_data:
            print("✗ 未能获取最新数据")
            return
        
        print(f"网站最新期号: {latest_data['期号']}")
        
        # 检查是否需要更新
        if local_latest:
            if int(latest_data['期号']) <= int(local_latest):
                print(f"✓ 本地数据已是最新，无需更新")
                return
        
        # 追加新数据
        print(f"\n正在更新期号 {latest_data['期号']} 的数据...")
        if self.append_to_csv(latest_data):
            print(f"✓ 成功更新！")
            print(f"  期号: {latest_data['期号']}")
            print(f"  开奖日期: {latest_data['开奖日期']}")
            print(f"  红球: {latest_data['红球1']} {latest_data['红球2']} {latest_data['红球3']} {latest_data['红球4']} {latest_data['红球5']} {latest_data['红球6']}")
            print(f"  蓝球: {latest_data['蓝球']}")
        else:
            print("✗ 更新失败")


def main():
    import sys
    
    # 支持命令行参数指定数据源
    source = 'auto'
    if len(sys.argv) > 1:
        source = sys.argv[1]
    
    updater = SSQLatestUpdater()
    updater.update(source=source)


if __name__ == '__main__':
    main()

