"""
测试API接口，查看实际返回的数据格式
"""
import requests
import json

# 测试接口
url = "https://m.78500.cn/kaijiang/ssq/"

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Content-Type': 'application/x-www-form-urlencoded'
}

# 测试第1页
print("=" * 60)
print("测试第1页数据")
print("=" * 60)

post_data = {
    'reqType': 'list',
    'reqVal': 'page=1'
}

try:
    response = requests.post(url, headers=headers, data=post_data, timeout=15)
    response.encoding = 'utf-8'
    
    print(f"状态码: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
    print(f"响应长度: {len(response.text)}")
    print()
    
    # 尝试解析JSON
    try:
        json_data = response.json()
        print("✓ 是JSON格式")
        print(f"数据类型: {type(json_data)}")
        print()
        
        if isinstance(json_data, dict):
            print("JSON键:")
            for key in json_data.keys():
                print(f"  - {key}")
            print()
            
            # 打印部分数据
            print("JSON内容（格式化后前2000字符）:")
            print(json.dumps(json_data, ensure_ascii=False, indent=2)[:2000])
            
        elif isinstance(json_data, list):
            print(f"是列表，包含 {len(json_data)} 个元素")
            if len(json_data) > 0:
                print()
                print("第一个元素:")
                print(json.dumps(json_data[0], ensure_ascii=False, indent=2))
                print()
                if len(json_data) > 1:
                    print("第二个元素:")
                    print(json.dumps(json_data[1], ensure_ascii=False, indent=2))
        else:
            print(f"JSON内容: {json_data}")
            
    except json.JSONDecodeError:
        print("✗ 不是JSON格式，是HTML")
        print()
        print("响应前1000字符:")
        print(response.text[:1000])
        print()
        print("查找包含'期号'的内容:")
        import re
        periods = re.findall(r'(19|20)\d{5}', response.text)
        print(f"找到 {len(periods)} 个期号模式: {periods[:10]}")
        
except Exception as e:
    print(f"请求失败: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("测试第2页数据")
print("=" * 60)

post_data2 = {
    'reqType': 'list',
    'reqVal': 'page=2'
}

try:
    response2 = requests.post(url, headers=headers, data=post_data2, timeout=15)
    response2.encoding = 'utf-8'
    
    print(f"状态码: {response2.status_code}")
    print(f"响应长度: {len(response2.text)}")
    
    try:
        json_data2 = response2.json()
        print("✓ 是JSON格式")
        if isinstance(json_data2, dict):
            print("JSON键:", list(json_data2.keys()))
        elif isinstance(json_data2, list):
            print(f"是列表，包含 {len(json_data2)} 个元素")
    except:
        print("✗ 不是JSON格式")
        
except Exception as e:
    print(f"请求失败: {e}")

