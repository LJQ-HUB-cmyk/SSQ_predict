"""
双色球历史数据获取脚本
从78500.cn获取历年双色球开奖数据
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
import os
import re


class SSQDataFetcher:
    def __init__(self):
        self.base_url = "https://m.78500.cn/kaijiang/ssq/"
        self.session = requests.Session()  # 使用session保持cookie
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-Requested-With': 'XMLHttpRequest',
            'Origin': 'https://m.78500.cn',
            'Referer': 'https://m.78500.cn/kaijiang/ssq/'
        }
        self.data_file = "ssq_history.csv"
        # 先访问主页获取cookie
        try:
            self.session.get(self.base_url, timeout=10)
        except:
            pass
    
    def fetch_from_page(self, page=1):
        """
        从78500.cn网站获取数据
        第一页：使用GET请求获取HTML页面
        其他页：使用POST请求获取JSON数据
        """
        try:
            url = self.base_url
            
            # 第一页使用GET请求获取HTML
            if page == 1:
                response = self.session.get(url, headers=self.headers, timeout=15)
                response.encoding = 'gb2312'  # 根据HTML的charset设置
                
                # 检查响应是否成功
                if response.status_code != 200:
                    print(f"  请求失败，状态码: {response.status_code}")
                    return None
                
                # 解析第一页的HTML内容
                soup = BeautifulSoup(response.text, 'html.parser')
                data = self._parse_first_page_html(soup)
                
                if data:
                    print(f"  从第一页HTML解析到 {len(data)} 条数据")
                    return data
                else:
                    print(f"  第一页HTML解析失败")
                    return None
            
            # 从第二页开始使用POST请求获取JSON数据
            else:
                post_data = {
                    'reqType': 'list',
                    'reqVal': '',  # 空值
                    'page': str(page)  # 页码直接作为page参数
                }
                
                response = self.session.post(url, headers=self.headers, data=post_data, timeout=15)
                response.encoding = 'utf-8'
                
                # 检查响应是否成功
                if response.status_code != 200:
                    print(f"  请求失败，状态码: {response.status_code}")
                    return None
                
                # 解析JSON数据
                content_type = response.headers.get('Content-Type', '')
                text_content = response.text.strip()
                
                # 检查是否是JSON响应
                if 'application/json' in content_type or text_content.startswith('{') or text_content.startswith('['):
                    try:
                        json_data = response.json()
                        
                        # 检查返回的页码是否匹配请求的页码
                        if isinstance(json_data, dict) and 'page' in json_data:
                            returned_page = json_data.get('page')
                            if returned_page != page:
                                print(f"  警告: 请求第{page}页，但返回第{returned_page}页")
                                # 如果返回的是第1页，说明分页不起作用
                                if returned_page == 1:
                                    return None
                        
                        parsed_data = self._parse_json_data(json_data)
                        if parsed_data:
                            return parsed_data
                    except Exception as e:
                        print(f"  JSON解析失败: {e}")
                        return None
                else:
                    print(f"  第{page}页返回的不是JSON格式")
                    return None
            
        except Exception as e:
            print(f"获取第{page}页数据失败: {e}")
            if page == 1:
                import traceback
                traceback.print_exc()
            return None
    
    def _parse_years_list(self, soup):
        """
        解析年份列表
        HTML格式：
        <div class="years" id="plist">
            <ul>
                <li data="2025">2025年</li>
                <li data="2024">2024年</li>
                ...
            </ul>
        </div>
        """
        years = []
        try:
            years_div = soup.find('div', class_='years', id='plist')
            if years_div:
                year_items = years_div.find_all('li')
                for item in year_items:
                    year_data = item.get('data', '')
                    if year_data and year_data.isdigit():
                        years.append(int(year_data))
                years.sort(reverse=True)  # 从新到旧排序
        except Exception as e:
            print(f"  解析年份列表失败: {e}")
        return years
    
    def _parse_first_page_html(self, soup):
        """
        解析第一页的HTML结构
        HTML格式：
        <section class="item">
            <a href="/kaijiang/ssq/2025127.html">
                <h3><strong>2025127期</strong><span>开奖时间 2025-11-04</span></h3>
                <p><i>03</i><i>09</i><i>15</i><i>17</i><i>19</i><i>28</i><b>03</b></p>
            </a>
        </section>
        """
        data = []
        try:
            # 查找所有的 <section class="item"> 元素
            items = soup.find_all('section', class_='item')
            
            for item in items:
                try:
                    # 提取期号：从 <strong>2025127期</strong> 中提取
                    strong_tag = item.find('strong')
                    if not strong_tag:
                        continue
                    
                    period_text = strong_tag.get_text().strip()
                    # 提取期号，格式可能是"2025127期"或"2025127"
                    period_match = re.search(r'((19|20)\d{5})', period_text)
                    if not period_match:
                        continue
                    period = period_match.group(1)
                    
                    # 提取日期：从 <span>开奖时间 2025-11-04</span> 中提取
                    date = ""
                    span_tag = item.find('span')
                    if span_tag:
                        date_text = span_tag.get_text().strip()
                        # 提取日期，格式可能是"开奖时间 2025-11-04"
                        date_match = re.search(r'(\d{4}-\d{1,2}-\d{1,2})', date_text)
                        if date_match:
                            date = date_match.group(1)
                    
                    # 提取号码：从 <p> 标签中提取
                    p_tag = item.find('p')
                    if not p_tag:
                        continue
                    
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
                    
                    # 验证：必须有6个不重复的红球和1个蓝球
                    if len(red_balls) == 6 and blue_ball is not None:
                        red_balls = sorted(red_balls)
                        data.append({
                            '期号': period,
                            '开奖日期': date,
                            '红球1': str(red_balls[0]),
                            '红球2': str(red_balls[1]),
                            '红球3': str(red_balls[2]),
                            '红球4': str(red_balls[3]),
                            '红球5': str(red_balls[4]),
                            '红球6': str(red_balls[5]),
                            '蓝球': str(blue_ball)
                        })
                except Exception as e:
                    continue
            
        except Exception as e:
            print(f"  解析第一页HTML失败: {e}")
        
        return data if data else None
    
    def _parse_table_row(self, cols):
        """解析表格行数据"""
        try:
            # 提取期号（跳过期号列，从其他列提取号码）
            period = cols[0].text.strip()
            # 验证期号格式：必须是年份开头的7位数字（如2025001）
            if not re.match(r'^(19|20)\d{5}$', period):
                return None
            
            date = cols[1].text.strip() if len(cols) > 1 else ""
            
            # 从第2列开始提取号码（跳过期号和日期列）
            all_numbers = []
            for col in cols[2:]:  # 跳过期号和日期列
                text = col.text.strip()
                # 更精确的数字提取：只提取1-33和1-16范围内的数字
                numbers = re.findall(r'\b([1-9]|[12]\d|3[0-3])\b', text)  # 1-33
                all_numbers.extend([int(n) for n in numbers])
                # 提取蓝球（1-16）
                blue_numbers = re.findall(r'\b([1-9]|1[0-6])\b', text)
                for num in blue_numbers:
                    if int(num) not in all_numbers:
                        all_numbers.append(int(num))
            
            # 分离红球（1-33）和蓝球（1-16）
            red_balls = []
            blue_ball = None
            
            for num in all_numbers:
                if 1 <= num <= 33:
                    if num not in red_balls:  # 去重
                        red_balls.append(num)
                elif 1 <= num <= 16:
                    if blue_ball is None:
                        blue_ball = num
            
            # 验证：必须有6个不重复的红球和1个蓝球
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
        except Exception as e:
            pass
        return None
    
    def _parse_text_item(self, item):
        """从文本元素中解析数据"""
        try:
            text = item.get_text()
            # 提取期号（必须是年份开头的7位数字）
            period_match = re.search(r'((19|20)\d{5})', text)
            if not period_match:
                return None
            
            period = period_match.group(1)
            
            # 提取日期
            date_match = re.search(r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})', text)
            date = date_match.group(1).replace('年', '-').replace('月', '-') if date_match else ""
            
            # 提取号码（更精确的正则）
            red_numbers = re.findall(r'\b([1-9]|[12]\d|3[0-3])\b', text)  # 1-33
            blue_numbers = re.findall(r'\b([1-9]|1[0-6])\b', text)  # 1-16
            
            # 去重并验证
            red_balls = sorted(list(set([int(n) for n in red_numbers if 1 <= int(n) <= 33])))
            blue_balls = list(set([int(n) for n in blue_numbers if 1 <= int(n) <= 16]))
            
            # 验证：必须有6个不重复的红球和至少1个蓝球
            if len(red_balls) == 6 and len(blue_balls) >= 1:
                return {
                    '期号': period,
                    '开奖日期': date,
                    '红球1': str(red_balls[0]),
                    '红球2': str(red_balls[1]),
                    '红球3': str(red_balls[2]),
                    '红球4': str(red_balls[3]),
                    '红球5': str(red_balls[4]),
                    '红球6': str(red_balls[5]),
                    '蓝球': str(blue_balls[0])
                }
        except Exception:
            pass
        return None
    
    def _parse_from_text(self, text):
        """从纯文本中解析数据"""
        data = []
        # 查找期号模式（必须是年份开头的7位数字）
        period_pattern = r'((19|20)\d{5})'
        date_pattern = r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})'
        
        # 查找所有期号
        periods = re.finditer(period_pattern, text)
        for period_match in periods:
            try:
                period = period_match.group(1)
                start_pos = period_match.start()
                end_pos = min(start_pos + 200, len(text))  # 在期号附近200字符内查找
                context = text[start_pos:end_pos]
                
                # 提取日期
                date_match = re.search(date_pattern, context)
                date = date_match.group(1).replace('年', '-').replace('月', '-') if date_match else ""
                
                # 提取号码（更精确）
                red_numbers = re.findall(r'\b([1-9]|[12]\d|3[0-3])\b', context)  # 1-33
                blue_numbers = re.findall(r'\b([1-9]|1[0-6])\b', context)  # 1-16
                
                # 去重并验证
                red_balls = sorted(list(set([int(n) for n in red_numbers if 1 <= int(n) <= 33])))
                blue_balls = list(set([int(n) for n in blue_numbers if 1 <= int(n) <= 16]))
                
                # 验证：必须有6个不重复的红球和至少1个蓝球
                if len(red_balls) == 6 and len(blue_balls) >= 1:
                    data.append({
                        '期号': period,
                        '开奖日期': date,
                        '红球1': str(red_balls[0]),
                        '红球2': str(red_balls[1]),
                        '红球3': str(red_balls[2]),
                        '红球4': str(red_balls[3]),
                        '红球5': str(red_balls[4]),
                        '红球6': str(red_balls[5]),
                        '蓝球': str(blue_balls[0])
                    })
            except Exception:
                continue
        
        return data
    
    def _parse_json_data(self, json_data):
        """解析JSON格式的数据"""
        data = []
        try:
            # 尝试不同的JSON结构
            if isinstance(json_data, dict):
                # 尝试常见的数据字段名
                items = None
                for key in ['data', 'list', 'results', 'items', 'rows', 'records', 'result', 'content']:
                    if key in json_data:
                        items = json_data[key]
                        break
                
                # 如果没找到，尝试查看所有键
                if items is None:
                    keys = list(json_data.keys())
                    print(f"  JSON字典键: {keys}")
                    # 如果只有一个键，尝试使用它
                    if len(keys) == 1:
                        items = json_data[keys[0]]
                    else:
                        items = [json_data]
                
                if items is None:
                    items = [json_data]
            elif isinstance(json_data, list):
                items = json_data
            else:
                print(f"  未知的JSON数据类型: {type(json_data)}")
                return None
            
            print(f"  找到 {len(items) if isinstance(items, list) else 1} 个数据项")
            
            parsed_count = 0
            for item in items:
                if isinstance(item, dict):
                    # 根据实际API格式解析：qishu=期号, result=结果数组, date=日期
                    # result数组：前6个是红球，最后1个是蓝球
                    period = str(item.get('qishu', item.get('qihao', item.get('期号', '')))).strip()
                    
                    # 验证期号格式
                    if not period or not re.match(r'^(19|20)\d{5}$', period):
                        continue
                    
                    date = str(item.get('date', item.get('开奖日期', ''))).strip()
                    
                    # 获取result数组
                    result = item.get('result', item.get('numbers', item.get('nums', [])))
                    
                    if not isinstance(result, list) or len(result) < 7:
                        # 尝试其他字段
                        for key in ['red', 'redBall', '红球', 'redBalls']:
                            if key in item:
                                red_val = item[key]
                                if isinstance(red_val, list):
                                    result = red_val
                                    break
                    
                    if not isinstance(result, list) or len(result) < 7:
                        continue
                    
                    # 提取红球（前6个）和蓝球（最后1个）
                    # 号码可能是字符串格式（如"06"）或数字格式
                    try:
                        red_balls = []
                        for i in range(6):
                            num = result[i]
                            if isinstance(num, str):
                                # 去除前导0，如"06" -> 6
                                num = int(num.lstrip('0') or '0')
                            else:
                                num = int(num)
                            if 1 <= num <= 33:
                                red_balls.append(num)
                        
                        # 提取蓝球（最后一个）
                        blue_val = result[6] if len(result) > 6 else None
                        if blue_val is None:
                            # 尝试从blue字段获取
                            blue_val = item.get('blue', item.get('blueBall', item.get('蓝球', '')))
                        
                        if isinstance(blue_val, str):
                            blue_ball = int(blue_val.lstrip('0') or '0')
                        elif isinstance(blue_val, (int, float)):
                            blue_ball = int(blue_val)
                        else:
                            continue
                        
                        # 验证：必须有6个不重复的红球和1个蓝球
                        red_balls = sorted(list(set(red_balls)))
                        if len(red_balls) == 6 and 1 <= blue_ball <= 16:
                            data.append({
                                '期号': period,
                                '开奖日期': date,
                                '红球1': str(red_balls[0]),
                                '红球2': str(red_balls[1]),
                                '红球3': str(red_balls[2]),
                                '红球4': str(red_balls[3]),
                                '红球5': str(red_balls[4]),
                                '红球6': str(red_balls[5]),
                                '蓝球': str(blue_ball)
                            })
                    except (ValueError, IndexError) as e:
                        if parsed_count < 3:
                            print(f"  解析号码失败: {e}")
                        continue
                    
                    parsed_count += 1
            
            print(f"  成功解析 {len(data)} 条有效数据")
        except Exception as e:
            print(f"  JSON解析异常: {e}")
            import traceback
            traceback.print_exc()
        
        return data if data else None
    
    def fetch_all_history(self):
        """
        获取所有历史数据（2003-2025年所有期号）
        """
        print("开始从 78500.cn 获取双色球历史数据...")
        all_data = []
        max_pages = 500  # 增加最大页数，确保能获取2003-2025年所有数据
        consecutive_failures = 0
        
        seen_periods = set()  # 用于检测重复数据
        page = 1
        
        # 先获取第一页，解析年份列表
        print("正在获取第 1 页数据...")
        response = self.session.get(self.base_url, headers=self.headers, timeout=15)
        response.encoding = 'gb2312'
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # 解析年份列表
            years_list = self._parse_years_list(soup)
            if years_list:
                print(f"  检测到年份范围: {min(years_list)}-{max(years_list)}年")
                expected_years = set(range(2003, 2026))  # 2003-2025年
                actual_years = set(years_list)
                missing_years = expected_years - actual_years
                if missing_years:
                    print(f"  警告: 缺少年份: {sorted(missing_years)}")
                else:
                    print(f"  ✓ 年份覆盖完整: 2003-2025年")
            
            # 解析第一页数据
            data = self._parse_first_page_html(soup)
            if data:
                new_periods = [item['期号'] for item in data]
                seen_periods.update(new_periods)
                all_data.extend(data)
                print(f"  成功获取 {len(data)} 期数据")
                page = 2  # 从第二页开始继续获取
            else:
                print(f"  第一页HTML解析失败")
        
        # 继续获取其他页面
        while page <= max_pages:
            print(f"正在获取第 {page} 页数据...")
            data = self.fetch_from_page(page)
            
            # 如果返回None，说明分页无效，停止获取
            if data is None:
                print(f"  API分页功能无效，停止获取")
                break
            
            # 如果有数据
            if data and len(data) > 0:
                # 检查是否有重复数据（说明分页不起作用）
                new_periods = [item['期号'] for item in data]
                duplicate_count = sum(1 for p in new_periods if p in seen_periods)
                
                if duplicate_count > 0:
                    print(f"  检测到 {duplicate_count} 条重复数据，API分页可能不起作用，停止获取")
                    break
                
                # 添加新的期号到已见集合
                seen_periods.update(new_periods)
                
                all_data.extend(data)
                consecutive_failures = 0
                print(f"  成功获取 {len(data)} 期数据，累计: {len(all_data)} 期")
                
                page += 1
                time.sleep(1)  # 避免请求过快
            else:
                # 数据为空或长度为0
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print(f"连续3页无数据，停止获取")
                    break
                page += 1
                time.sleep(1)
        
        if all_data:
            df = pd.DataFrame(all_data)
            # 数据验证和清洗
            # 1. 验证期号格式
            valid_periods = df['期号'].str.match(r'^(19|20)\d{5}$')
            df = df[valid_periods].copy()
            
            # 2. 验证红球不重复且范围正确
            red_cols = ['红球1', '红球2', '红球3', '红球4', '红球5', '红球6']
            for col in red_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['蓝球'] = pd.to_numeric(df['蓝球'], errors='coerce')
            
            # 验证红球范围（1-33）且不重复
            def validate_red_balls(row):
                reds = [int(row[c]) for c in red_cols]
                if len(set(reds)) != 6:  # 必须6个不重复
                    return False
                if any(r < 1 or r > 33 for r in reds):
                    return False
                return True
            
            def validate_blue_ball(row):
                blue = int(row['蓝球'])
                return 1 <= blue <= 16
            
            valid_red = df.apply(validate_red_balls, axis=1)
            valid_blue = df.apply(validate_blue_ball, axis=1)
            df = df[valid_red & valid_blue].copy()
            
            # 3. 去重
            df = df.drop_duplicates(subset=['期号'], keep='first')
            
            # 4. 按期号排序（最新的在前）
            df = df.sort_values('期号', ascending=False)
            
            # 5. 验证年份覆盖（确保2003-2025年所有年份都有数据）
            df['年份'] = df['期号'].str[:4].astype(int)
            years_in_data = set(df['年份'].unique())
            expected_years = set(range(2003, 2026))  # 2003-2025年
            missing_years = expected_years - years_in_data
            
            if missing_years:
                print(f"⚠️  警告: 缺少年份数据: {sorted(missing_years)}")
                print(f"   当前数据覆盖年份: {sorted(years_in_data)}")
            else:
                print(f"✓ 年份覆盖完整: 2003-2025年所有年份都有数据")
            
            # 统计每年期数
            year_counts = df.groupby('年份')['期号'].count().sort_index()
            print(f"\n各年份期数统计:")
            for year in sorted(years_in_data):
                count = year_counts[year]
                print(f"  {year}年: {count}期")
            
            # 删除临时添加的年份列
            df = df.drop(columns=['年份'])
            
            # 6. 保存
            df.to_csv(self.data_file, index=False, encoding='utf-8-sig')
            print(f"\n✓ 成功获取 {len(df)} 期有效数据，已保存到 {self.data_file}")
            
            if len(df) == 0:
                print("警告: 没有有效数据被获取")
            
            return df
        else:
            print("未能获取数据")
            return None
    
    def create_sample_data(self):
        """
        如果无法从网络获取，创建示例数据文件
        实际使用时应该替换为真实数据
        """
        print("创建示例数据文件...")
        sample_data = []
        # 生成一些示例数据供测试使用
        np.random.seed(42)  # 设置随机种子以保证可重现性
        for i in range(2000, 2024):
            for j in range(1, 154):  # 每年约154期
                period = f"{i}{j:03d}"
                # 生成6个不重复的红球（1-33）
                red_balls = sorted(np.random.choice(range(1, 34), size=6, replace=False).tolist())
                # 生成1个蓝球（1-16）
                blue_ball = np.random.randint(1, 17)
                
                sample_data.append({
                    '期号': period,
                    '开奖日期': f"{i}-01-01",
                    '红球1': str(red_balls[0]),
                    '红球2': str(red_balls[1]),
                    '红球3': str(red_balls[2]),
                    '红球4': str(red_balls[3]),
                    '红球5': str(red_balls[4]),
                    '红球6': str(red_balls[5]),
                    '蓝球': str(blue_ball)
                })
        
        df = pd.DataFrame(sample_data)
        df.to_csv(self.data_file, index=False, encoding='utf-8-sig')
        print(f"示例数据已保存到 {self.data_file}")
        return df


if __name__ == "__main__":
    fetcher = SSQDataFetcher()
    
    # 尝试从网络获取数据
    df = fetcher.fetch_all_history()
    
    # 如果获取失败或数据量太少，使用示例数据
    if df is None or len(df) == 0:
        print("\n网络获取失败，使用示例数据...")
        df = fetcher.create_sample_data()
    elif len(df) < 100:
        print(f"\n⚠️ 警告: 只获取到 {len(df)} 期数据，对于LSTM训练来说太少了（建议至少100期）")
        print("是否使用示例数据？(脚本会自动生成)")
        
        # 检查是否已有足够的历史数据（如果之前运行过）
        if os.path.exists(fetcher.data_file):
            with open(fetcher.data_file, 'r', encoding='utf-8-sig') as f:
                existing_lines = len(f.readlines()) - 1
            
            if existing_lines < 100:
                print("自动切换到示例数据以支持模型训练...")
                df = fetcher.create_sample_data()
            else:
                print(f"使用现有的 {existing_lines} 期数据进行训练")
        else:
            print("自动切换到示例数据以支持模型训练...")
            df = fetcher.create_sample_data()
    
    if df is not None:
        print(f"\n✓ 最终数据量: {len(df)} 期")
        print(f"✓ 数据已保存到: {fetcher.data_file}")
        print(f"✓ 可以开始数据预处理和模型训练")

