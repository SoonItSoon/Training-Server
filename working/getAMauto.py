# 재난문자 가져오기 ver.2
# 자동으로 재난문자 다 가져오기

from selenium import webdriver
# import pymysql
import csv
import time
import re

CSV_FILE = "./AM_csv.csv"
# 가장 최근의 일련번호를 갖고있는 변수
lastMID = -1
# 실패한 일련번호를 저장하는 list
msgList = []
failList = []
# 국민재난안전포털 재난문자 URL
URL = "http://www.safekorea.go.kr/idsiSFK/neo/sfk/cs/sfc/dis/disasterMsgView.jsp?menuSeq=679"

# webdriver 설정
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('headless')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('lang=ko_KR')

# csv 파일에서 
def getLastMID():
    global CSV_FILE
    file = open(CSV_FILE, "r", encoding="utf-8")
    reader = csv.reader(file)
    lastMID = -1
    for line in reader:
        lastMID = line[0]
    now_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    log_default = f"{now_date} [S_getMsgData]"
    print(f"{log_default} getLastMID() : {lastMID}")
    return int(lastMID) + 1


def getSender(msg):
    search = re.search("\[(.+?)\]", msg)
    if search:
        sender = search.group(1).split(",")[0].split()[0]
        sender = re.sub("[0-9①-⑮\[\- ]", "", sender)
        temp = sender.split("(")
        if len(temp[0]) > 0:
            sender = temp[0]
        elif len(temp) > 1:
            sender = temp[1]
        else:
            sender = ""
    else:
        sender = ""
    return sender


def saveMsg2DB(msgList):
    # DB 접속
    AlertMsgDB = pymysql.connect(
        user='kyeol',
        passwd='hee',
        host='127.0.0.1',
        db='AlertMsgDB',
        charset='utf8'
    )
    # mid, send_date, msg, send_location, sender, disaster
    insertSQL = "INSERT INTO AMtest VALUES (%s, %s, %s, %s, %s, %s)"
    cursor = AlertMsgDB.cursor(pymysql.cursors.DictCursor)
    cursor.executemany(insertSQL, msgList)
    AlertMsgDB.commit()
    cursor.close()
    AlertMsgDB.close()


def saveMsg2CSV(msgList):
    # 재난문자 데이터 csv로 저장
    global CSV_FILE
    file = open(CSV_FILE, mode="a", newline="", encoding='utf-8')
    writer = csv.writer(file)

    for info in msgList:
        writer.writerow(info)

    file.close()


def getMsgData():
    # time.sleep(10)
    now_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    log_default = f"{now_date} [S_getMsgData]"
    global lastMID
    if lastMID == -1:
        lastMID = getLastMID()
    index = lastMID
    # 브라우저 생성
    global chrome_options
    browser = webdriver.Chrome(executable_path="./chromedriver.exe", options=chrome_options)
    browser.get(URL)
    # 재난문자를 임시 저장하는 list
    msgList = []
    pdList = []
    received = True
    failCnt = 0
    while received == True:
    # for index in [24410, 24518, 24913, 24999, 25076, 26266, 27294, 27299, 31993, 34430, 35057, 44665, 45733, 47479, 48758, 50010, 53035, 54124, 55945, 67578]:
        nextBtn = browser.find_element_by_id("bbs_gubun")
        browser.execute_script("arguments[0].href = arguments[1]", nextBtn, f"javascript:articleDtl('63','{index}');")
        nextBtn.send_keys('\n')
        # 페이지가 로딩 중인 경우를 고려하여 정보가 표시되지 않을 경우 대기 후 재시도
        sendTime = browser.find_element_by_id("sj").text
        if len(sendTime) == 0:
            time.sleep(2)
        sendTime = browser.find_element_by_id("sj").text.split()

        global failList
        # 정보가 없는 경우 failList에 추가
        if len(sendTime) == 0:
            failList.append(index)
            failCnt += 1
            print(f"!!!{index}번 재난문자 실패({lastMID}, {failCnt})!!!")
            if failCnt >= 3:
                tempList = failList[-3:]
                failList = [x for x in failList if x not in tempList]
                received = False
                continue
        # 정보가 있을 경우 추출 후 msgList에 저장
        else:
            failCnt = 0
            context = browser.find_element_by_id("cn").text.split("\n-송출지역-\n")
            # 전 지역으로 발송된 재난문자 (송출 지역 없음)
            if len(context) < 2:
                context.append("")
            # 재난문자 데이터 가공
            send_date = sendTime[0].replace('/', '-') + " " + sendTime[1]
            msg = context[0].replace("\n", " ")
            send_location = context[1].replace("\n", ",")
            sender = getSender(msg)
            disasterType = 0
            # 재난문자 데이터 임시 저장
            # mid, send_date, msg, send_location, sender, disaster
            msgList.append([index, send_date, msg, send_location, sender, disasterType])
            print(f"[S_getMsgData] SUCCESS loading {{mid: {index}, send_date: {send_date}, msg: {msg}, send_location : {send_location}, sender: {sender}, disaster: {disasterType}}} ({len(msgList)})")
            lastMID = index + 1
            print(f"!!!{index}번 재난문자 성공!!!")
        index += 1
    browser.quit()

    # 추출한 재난문자가 있다면 AlertMsg.AM에 저장
    if len(msgList) > 0:
        # saveMsg2DB(msgList)
        saveMsg2CSV(msgList)
        print(f"!!!재난문자{len(msgList)}개 저장 완료({lastMID})!!!")
        msgList.clear()

    print(f"!!!실패 리스트 : {failList}")


getMsgData()
