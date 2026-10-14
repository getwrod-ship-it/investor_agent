# -*- coding: utf-8 -*-
"""
litemall 业务数据批量生成器
- 600 个用户 + 5 万订单(近18个月,含双11/618高峰) + 订单商品 + 优惠券领取/使用 + 售后单
- 金额公式与表注释严格一致:
    order_price  = goods_price + freight_price - coupon_price
    actual_price = order_price - integral_price
- 订单状态: 101未付款 102用户取消 103超时取消 201已付款 202退款申请 203已退款
            301已发货 401确认收货 402超期自动收货
- 固定随机种子,可重复执行得到相同结果。重跑前请先清空相关表(脚本不删除已有数据)。
"""
import random
from datetime import datetime, timedelta

import pymysql

# ---------------------------------------------------------------------------
DB = dict(host="127.0.0.1", user="root", password="123456",
          database="litemall", charset="utf8mb4")
N_USERS = 600
N_ORDERS = 50_000
N_EXTRA_COUPONS = 6_000          # 未使用/已过期的优惠券领取记录
SEED = 42
NOW = datetime(2026, 10, 12, 12, 0, 0)   # 以该时刻为"当前时间"
SPAN_DAYS = 540                  # 近 18 个月

random.seed(SEED)

# ---------------------------------------------------------------------------
# 基础语料
SURNAMES = "王李张刘陈杨赵黄周吴徐孙马朱胡郭何高林郑谢罗宋唐韩冯于董萧程曹袁邓许傅沈曾彭吕"
GIVEN = ["伟", "芳", "娜", "敏", "静", "磊", "军", "洋", "勇", "艳", "杰", "涛", "明", "超",
         "秀英", "建国", "桂英", "鑫", "浩然", "子轩", "雨萱", "思远", "梦琪", "梓涵",
         "小燕", "志强", "文博", "嘉怡", "俊杰", "丽华", "建国", "国庆", "小红"]
CITIES = ["北京市朝阳区", "上海市浦东新区", "广州市天河区", "深圳市南山区", "杭州市西湖区",
          "成都市武侯区", "武汉市洪山区", "南京市鼓楼区", "重庆市渝北区", "西安市雁塔区",
          "苏州市工业园区", "长沙市岳麓区", "郑州市金水区", "青岛市市南区", "合肥市蜀山区"]
STREETS = ["科技路12号", "中山大道88号", "人民东路45弄", "解放北路156号", "文化路9号",
           "金融街3号楼", "滨江大道77号", "建设大街201号", "新华路65号", "幸福小区6栋2单元"]
MESSAGES = ["尽快发货", "请包装好,谢谢", "到货请电话联系", "需要电子发票", "勿放快递柜",
            "选顺丰,加急", "送人,请检查外观", ""]
REASONS = ["七天无理由退换货", "不想要了", "商品与描述不符", "质量问题", "发货太慢", "尺码不合适"]
REFUND_TYPES = ["原路退回", "微信退款", "支付宝退款"]
SHIP_CHANNELS = ["顺丰速运", "中通快递", "圆通速递", "韵达快递", "京东物流"]

# 近 18 个月销量权重(大促月放大):双11、双12、618、99大促
MONTH_BOOST = {(2025, 11): 2.4, (2025, 12): 1.3, (2026, 6): 1.9,
               (2026, 9): 1.2, (2026, 10): 1.1}
# 下单时段权重
HOUR_W = {0: .2, 1: .1, 2: .1, 3: .1, 4: .1, 5: .1, 6: .2, 7: .5, 8: .7, 9: .9,
          10: 1.3, 11: 1.2, 12: 1.0, 13: .8, 14: .9, 15: 1.1, 16: 1.0, 17: 1.0,
          18: .9, 19: .9, 20: 1.4, 21: 1.5, 22: 1.0, 23: .5}

FAKE_PWD = "$2a$10$VKiLeL5V0lSvkZ1kFhU6BOdTNd1yVZaSQxGmF3fOqPtXRTxS8N/eW"


def pick(rng_weight: dict) -> int:
    """按权重字典挑一个 key"""
    ks, ws = zip(*rng_weight.items())
    return random.choices(ks, weights=ws)[0]


def rand_name():
    return random.choice(SURNAMES) + random.choice(GIVEN)


def rand_mobile():
    return "1" + random.choice("345789") + "".join(random.choices("0123456789", k=9))


def main():
    conn = pymysql.connect(**DB, autocommit=False)
    cur = conn.cursor()

    # ---------------- 清理上一次生成的数据(保证可重跑,不碰演示数据) ----------------
    cur.execute("""DELETE cu FROM litemall_coupon_user cu
                   JOIN litemall_user u ON cu.user_id = u.id
                   WHERE u.username REGEXP '^user_[0-9]{4}$'""")
    cur.execute("DELETE FROM litemall_aftersale WHERE order_id IN (SELECT id FROM litemall_order)")
    cur.execute("TRUNCATE TABLE litemall_order")
    cur.execute("TRUNCATE TABLE litemall_order_goods")
    cur.execute("DELETE FROM litemall_user WHERE username REGEXP '^user_[0-9]{4}$'")
    conn.commit()

    # ---------------- 商品/货品(真实数据,保证价格和名称一致) ----------------
    cur.execute("""
        SELECT p.id, p.goods_id, p.specifications, p.price, p.url,
               g.name, g.goods_sn, g.pic_url
        FROM litemall_goods_product p JOIN litemall_goods g ON g.id = p.goods_id
        WHERE p.deleted = 0 AND g.deleted = 0""")
    products = cur.fetchall()
    # 幂律 popularity:前 20% 的货品占约 60% 销量
    weights = [1 / (i + 1) ** 0.7 for i in range(len(products))]

    # ---------------- 用户 ----------------
    users = []
    for i in range(1, N_USERS + 1):
        reg = NOW - timedelta(days=random.uniform(0, SPAN_DAYS))
        users.append(dict(
            nickname=rand_name() + str(random.randint(1, 999)),
            gender=random.choices([0, 1, 2], weights=[.1, .55, .35])[0],
            birthday=(datetime(1970, 1, 1) + timedelta(days=random.randint(0, 12000))).date(),
            level=random.choices([0, 1, 2], weights=[.9, .08, .02])[0],
            mobile=rand_mobile(), add=reg))
    cur.executemany("""INSERT INTO litemall_user
        (username,password,gender,birthday,last_login_time,last_login_ip,user_level,
         nickname,mobile,status,add_time,update_time,deleted)
        VALUES (%s,%s,%s,%s,NOW(),%s,%s,%s,%s,0,%s,NOW(),0)""",
        [(f"user_{i:04d}", FAKE_PWD, u["gender"], u["birthday"],
          f"192.168.{random.randint(0,255)}.{random.randint(1,254)}", u["level"],
          u["nickname"], u["mobile"], u["add"]) for i, u in enumerate(users, 1)])
    cur.execute("SELECT id FROM litemall_user WHERE username REGEXP '^user_[0-9]{4}$' ORDER BY id")
    real_ids = [r[0] for r in cur.fetchall()]
    assert len(real_ids) == len(users)
    for u, rid in zip(users, real_ids):
        u["id"] = rid
    print(f"用户: {len(users)}")

    # ---------------- 月份分配(含大促权重) ----------------
    months = []
    start = datetime(NOW.year, NOW.month, NOW.day) - timedelta(days=SPAN_DAYS)
    y, m = start.year, start.month
    cur_m = (y, m)
    while cur_m <= (NOW.year, NOW.month):
        boost = MONTH_BOOST.get(cur_m, 1.0)
        months.append((cur_m, boost))
        m += 1
        if m == 13:
            y, m = y + 1, 1
        cur_m = (y, m)
    month_w = [b for _, b in months]
    assert months, "no months"

    COUPONS = {1: 5.0, 2: 10.0, 3: 10.0, 8: 15.0}   # id -> 券面额(满99可用)

    orders, order_goods, coupon_used, aftersales = [], [], [], []
    hot_users = random.sample(users, k=int(N_USERS * 0.2))  # 20% 高频用户贡献 80% 订单

    for i in range(N_ORDERS):
        # --- 时间 ---
        yy, mm = random.choices([mp for mp, _ in months], weights=month_w)[0]
        day = random.choices(range(1, 28), weights=[1.3 if dd in (1, 11, 18) else 1.0
                                                    for dd in range(1, 28)])[0]
        add = datetime(yy, mm, day, pick(HOUR_W), random.randint(0, 59),
                       random.randint(0, 59))
        if add > NOW:
            add = NOW - timedelta(minutes=random.randint(5, 800))
        age = (NOW - add).days

        u = random.choice(hot_users) if random.random() < .8 else random.choice(users)

        # --- 商品明细 ---
        n_items = random.choices([1, 2, 3], weights=[.55, .3, .15])[0]
        items = random.choices(products, weights=weights, k=n_items)
        goods_price = 0.0
        for it in items:
            qty = random.choices([1, 2, 3], weights=[.8, .15, .05])[0]
            price = float(it[3])
            goods_price += round(price * qty, 2)
        goods_price = round(goods_price, 2)

        freight = 0.0 if goods_price >= 99 else 8.0
        coupon_id, coupon_price = None, 0.0
        if goods_price >= 99 and random.random() < .15:
            coupon_id = random.choice(list(COUPONS))
            coupon_price = COUPONS[coupon_id]
        integral = 0.0 if random.random() < .9 else round(random.uniform(.5, 5), 2)
        order_price = round(goods_price + freight - coupon_price, 2)
        actual_price = round(order_price - integral, 2)

        # --- 状态(按订单年龄演化) ---
        if age < 1:
            status = random.choices([101, 201], weights=[.2, .8])[0]
        elif age < 3:
            status = random.choices([201, 301, 102], weights=[.25, .55, .2])[0]
        elif age < 15:
            status = random.choices([301, 401, 202, 103], weights=[.35, .4, .1, .15])[0]
        else:
            status = random.choices([401, 402, 203, 103], weights=[.35, .52, .09, .04])[0]

        pay_time = ship_time = confirm_time = end_time = refund_time = None
        refund_amount = refund_type = refund_content = None
        aftersale_status = 0
        end_events = [add]

        if status != 101:                                  # 已支付
            pay_time = add + timedelta(minutes=random.randint(1, 30))
            end_events.append(pay_time)
        if status in (301, 401, 402):                      # 已发货及之后
            ship_time = pay_time + timedelta(hours=random.uniform(6, 48))
            end_events.append(ship_time)
        if status in (401, 402):                           # 已收货
            confirm_time = ship_time + timedelta(days=random.uniform(2, 7))
            end_events.append(confirm_time)
        if status in (102, 103):                           # 取消
            end_time = add + timedelta(minutes=random.randint(10, 120))
            end_events.append(end_time)
        if status in (202, 203):                           # 售后/退款
            apply_t = (add if status == 202 else
                       confirm_time or ship_time or pay_time) + timedelta(hours=random.uniform(2, 72))
            refund_amount = round(actual_price * (1 if random.random() < .85
                                                  else random.uniform(.5, .9)), 2)
            refund_type = random.choice(REFUND_TYPES)
            refund_content = random.choice(REASONS)
            aftersale_status = 1 if status == 202 else 3
            if status == 203:
                refund_time = apply_t + timedelta(days=random.uniform(1, 4))
                end_time = refund_time
                end_events.append(refund_time)
            aftersales.append(dict(order_idx=i, user_id=u["id"], type=0 if status == 202 else 1,
                                   reason=refund_content, amount=refund_amount,
                                   status=aftersale_status, apply_t=apply_t,
                                   handle_t=refund_time))  # 202(已申请)尚未处理,handle_time 为空

        deleted = 1 if random.random() < .03 else 0
        if deleted and end_time is None:
            end_time = end_events[-1]
        update_time = max(end_events)
        comments = 0
        if status in (401, 402) and random.random() < .4:
            comments = n_items                             # 仍有 n 件待评价

        orders.append((
            u["id"], f"{add.strftime('%Y%m%d%H%M%S')}{random.randint(100000, 999999)}",
            status, aftersale_status, rand_name(), rand_mobile(),
            random.choice(CITIES) + random.choice(STREETS), random.choice(MESSAGES),
            goods_price, freight, coupon_price, integral, 0.0,
            order_price, actual_price,
            None if status == 101 else f"pay{random.getrandbits(48):012x}",
            pay_time,
            None if status in (101, 201, 102, 103, 202) else random.getrandbits(32),
            None if status in (301, 401, 402) else random.choice(SHIP_CHANNELS),
            ship_time, refund_amount, refund_type, refund_content, refund_time,
            confirm_time, comments, end_time, add, update_time, deleted))
        if coupon_id:
            coupon_used.append(dict(user_id=u["id"], coupon_id=coupon_id,
                                    used_time=pay_time, start=add - timedelta(days=15),
                                    end=add + timedelta(days=15),
                                    got=add - timedelta(days=random.randint(1, 10)),
                                    upd=update_time, order_idx=i))

        for it in items:
            order_goods.append((
                i,  # 所属订单在 orders 列表中的下标,回填时映射为真实 order_id
                it[1], it[5], it[6], it[0],
                random.choices([1, 2, 3], weights=[.8, .15, .05])[0],
                float(it[3]), it[2], it[7] or "",
                -1 if status in (202, 203) else 0,
                add, update_time, deleted))
    print(f"订单: {len(orders)}  明细: {len(order_goods)}  用券: {len(coupon_used)}  售后: {len(aftersales)}")

    # ---------------- 写入订单,利用 AUTO_INCREMENT 连续性回填 order_id ----------------
    cur.execute("SELECT COUNT(*) FROM litemall_order")
    assert cur.fetchone()[0] == 0, "litemall_order 非空,为避免 id 错位请先清空表"
    COLS = ("user_id,order_sn,order_status,aftersale_status,consignee,mobile,address,message,"
            "goods_price,freight_price,coupon_price,integral_price,groupon_price,order_price,"
            "actual_price,pay_id,pay_time,ship_sn,ship_channel,ship_time,refund_amount,"
            "refund_type,refund_content,refund_time,confirm_time,comments,end_time,add_time,"
            "update_time,deleted").split(",")
    ROW = "%s," * 29 + "%s"
    for s in range(0, len(orders), 1000):
        cur.executemany(
            f"INSERT INTO litemall_order ({','.join(COLS)}) VALUES ({ROW})", orders[s:s + 1000])
    # 单表独占写入,AUTO_INCREMENT 按插入顺序递增,可用行序回填 order_id
    cur.execute("SELECT id FROM litemall_order ORDER BY id")
    ids = [r[0] for r in cur.fetchall()]
    assert len(ids) == len(orders), f"id 数不符 {len(ids)} vs {len(orders)}"
    order_id_by_idx = {i: oid for i, oid in enumerate(ids)}
    og_rows, cu_rows, as_rows = [], [], []
    for r in order_goods:
        og_rows.append((order_id_by_idx[r[0]],) + tuple(r[1:]))
    for c in coupon_used:
        cu_rows.append((c["user_id"], c["coupon_id"], 1, c["used_time"], c["start"],
                        c["end"], order_id_by_idx[c["order_idx"]], c["got"], c["upd"]))
    for a in aftersales:
        as_rows.append(("AS" + random.getrandbits(40).__format__("010x"),
                        order_id_by_idx[a["order_idx"]], a["user_id"], a["type"],
                        a["reason"], a["amount"], "[]", "", a["status"],
                        a["handle_t"], a["apply_t"], a["handle_t"] or a["apply_t"], 0))

    for s in range(0, len(og_rows), 2000):
        cur.executemany("""INSERT INTO litemall_order_goods
            (order_id,goods_id,goods_name,goods_sn,product_id,number,price,specifications,
             pic_url,comment,add_time,update_time,deleted) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            og_rows[s:s + 2000])
    for r in cu_rows:
        pass
    # 已用券
    cur.executemany("""INSERT INTO litemall_coupon_user
        (user_id,coupon_id,status,used_time,start_time,end_time,order_id,add_time,update_time)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""", cu_rows)

    # 未使用/已过期券
    extra = []
    for _ in range(N_EXTRA_COUPONS):
        u = random.choice(users)
        cid = random.choice(list(COUPONS))
        st = random.choices([0, 2], weights=[.4, .6])[0]
        stt = NOW - timedelta(days=random.randint(1, 120))
        expired = st == 2
        extra.append((u["id"], cid, st, None,
                      stt, stt + timedelta(days=30), None, stt, stt))
    cur.executemany("""INSERT INTO litemall_coupon_user
        (user_id,coupon_id,status,used_time,start_time,end_time,order_id,add_time,update_time)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""", extra)

    cur.executemany("""INSERT INTO litemall_aftersale
        (aftersale_sn,order_id,user_id,type,reason,amount,pictures,comment,status,
         handle_time,add_time,update_time,deleted)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", as_rows)

    conn.commit()

    cur.execute("""SELECT 'orders' t,COUNT(*) FROM litemall_order UNION ALL
                   SELECT 'order_goods',COUNT(*) FROM litemall_order_goods UNION ALL
                   SELECT 'coupon_user',COUNT(*) FROM litemall_coupon_user UNION ALL
                   SELECT 'aftersale',COUNT(*) FROM litemall_aftersale UNION ALL
                   SELECT 'users',COUNT(*) FROM litemall_user""")
    for t, c in cur.fetchall():
        print(f"{t}: {c}")
    conn.close()
    print("DONE")


if __name__ == "__main__":
    main()
