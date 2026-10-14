# 02 指标口径与标准 SQL

电商分析的标准指标在本库中的定义与写法。**所有 SQL 默认补 deleted=0**。基础口径:有效订单 = `order_status IN (201,301,401,402,403)`(已支付、未退款、未取消);退款中/已退款(202/203)不计入销售额但单独统计退款指标。

## 有效订单口径(一切销售指标的前提)

有效订单 = 已支付且未退款未取消:`order_status IN (201,301,401,402,403) AND deleted=0`。101(未付款)、102/103/104(取消)、202/203(退款)全部排除。这是"销售额、订单量、客单价"的默认过滤条件;仅当用户明确说"含未付款/GMV口径"时才放宽。

## GMV 与销售额的区别与写法

GMV(商品交易总额)= 统计期内**所有下单**(含未付款、退款前)的金额合计,行业上反映交易规模:`SELECT SUM(actual_price) FROM litemall_order WHERE deleted=0 AND 时间条件`。销售额(确认成交口径)= 有效订单实付合计,反映真实经营:`SELECT SUM(actual_price) FROM litemall_order WHERE order_status IN (201,301,401,402,403) AND deleted=0 AND 时间条件`。用户说"流水/规模"倾向 GMV,说"收入/卖了多少钱"用销售额口径。金额单位:元。

## 净销售回款(销售额−退款)

净回款 = 销售额 − 退款金额:`SELECT SUM(actual_price) - COALESCE(SUM(refund_amount),0) FROM litemall_order WHERE deleted=0 AND (order_status IN (201,301,401,402,403) OR order_status=203)`。注意 203 的实付已在销售额口径外,需把 203 的 actual_price 加回再减 refund_amount,或直接用上面写法。它仍不是利润(无成本数据)。

## 客单价(AOV)

客单价 = 有效订单销售额 ÷ 有效订单数,反映单次成交金额:`SELECT SUM(actual_price)/COUNT(*) FROM litemall_order WHERE order_status IN (201,301,401,402,403) AND deleted=0`。本库当前约 1,249 元/单(家居大件为主,客单较高)。区分"人均消费(ARPU)"= 销售额 ÷ 去重下单用户数,恒小于等于客单价(有复购)。

## 复购率

复购率 = 统计期内下单 ≥2 次的用户数 ÷ 下单用户数。标准 SQL:`SELECT COUNT(*) FROM (SELECT user_id FROM litemall_order WHERE order_status IN (201,301,401,402,403) AND deleted=0 AND 时间条件 GROUP BY user_id HAVING COUNT(*)>=2) t` 除以 `COUNT(DISTINCT user_id)`(同条件)。业务含义:衡量回头客比例与商品粘性;家居百货复购偏低属正常。

## 退款率(单量口径与金额口径)

单量口径 = 退款订单数 ÷ 总下单量:`SELECT SUM(order_status IN (202,203))/COUNT(*) FROM litemall_order WHERE deleted=0`。金额口径 = `SUM(refund_amount)/SUM(actual_price)`。业务含义:商品质量与售后压力;本库退款率约 8.6%(4,340/50,000),主要原因分布见 litemall_aftersale.reason。

## 动销率

动销率 = 统计期内在售且有成交的商品数 ÷ 在售商品总数,衡量商品结构效率:`SELECT COUNT(DISTINCT og.goods_id) / (SELECT COUNT(*) FROM litemall_goods WHERE is_on_sale=1 AND deleted=0) FROM litemall_order_goods og JOIN litemall_order o ON o.id=og.order_id WHERE og.deleted=0 AND o.order_status IN (201,301,401,402,403)`。分母当前 238 款。

## 优惠券核销率

核销率 = 已使用券 ÷ 领取券:`SELECT SUM(status=1)/COUNT(*) FROM litemall_coupon_user WHERE deleted=0`。当前约 52%(6,459 已用 / 12,459 领取)。延伸:各券种的核销率按 coupon_id 分组;过期作废率 = status=2 占比。

## 发货时效(履约速度)

平均发货时长 = 支付到发货的小时数:`SELECT AVG(TIMESTAMPDIFF(HOUR, pay_time, ship_time)) FROM litemall_order WHERE order_status IN (301,401,402) AND deleted=0 AND pay_time IS NOT NULL AND ship_time IS NOT NULL`。业务含义:履约能力;可按快递公司(ship_channel)分维度对比。

## 退款处理时长

平均退款处理时长 = 售后申请到处理完成:`SELECT AVG(TIMESTAMPDIFF(HOUR, add_time, handle_time)) FROM litemall_aftersale WHERE status=3 AND deleted=0`。仅对已完成的售后(status=3)计算,处理中(status=1)无 handle_time。

## 新客成交占比

新客 = 注册当月即产生首单的用户。占比 = 新客订单数 ÷ 有效订单数。SQL 思路:用户表 add_time 所在月 与 该用户 MIN(order.add_time) 所在月相等的订单占比。业务含义:增长依赖拉新还是留存。

## 库存与售罄风险

低库存预警 = 在售货品库存低于阈值:`SELECT p.id, g.name, p.specifications, p.number FROM litemall_goods_product p JOIN litemall_goods g ON g.id=p.goods_id WHERE p.deleted=0 AND g.is_on_sale=1 AND p.number < 10 ORDER BY p.number`。行业惯例阈值 10~50,按销量速度调整。
