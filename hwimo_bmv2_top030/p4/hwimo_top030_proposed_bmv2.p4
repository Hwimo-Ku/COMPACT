#include <core.p4>
#include <v1model.p4>

const bit<16> TYPE_IPV4 = 0x0800;
const bit<8> IP_PROTO_HWIMO = 200;

header ethernet_t {
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16> etherType;
}

header ipv4_t {
    bit<4>  version;
    bit<4>  ihl;
    bit<8>  diffserv;
    bit<16> totalLen;
    bit<16> identification;
    bit<3>  flags;
    bit<13> fragOffset;
    bit<8>  ttl;
    bit<8>  protocol;
    bit<16> hdrChecksum;
    bit<32> srcAddr;
    bit<32> dstAddr;
}

header hwimo_t {
    bit<32> sample_id;
}

header features_t {
    bit<153> bits;
    bit<7>   pad;
}

header prediction_t {
    bit<1> pred_task1;
    bit<1> pred_task2;
    bit<3> pred_task3;
    bit<3> pred_task4;
    bit<2> pred_task5;
    bit<2> pred_task6;
    bit<2> pred_task7;
    bit<2> pad;
}

header tcp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4>  dataOffset;
    bit<3>  res;
    bit<3>  ecn;
    bit<6>  ctrl;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgentPtr;
}

struct headers_t {
    ethernet_t   ethernet;
    ipv4_t       ipv4;
    hwimo_t      hwimo;
    features_t   features;
    prediction_t prediction;
    tcp_t        tcp;
}

struct metadata_t {
    bit<30> key_1;
    bit<30> key_2;
    bit<30> key_3;
    bit<30> key_4;
    bit<30> key_5_6_7;
}

parser MyParser(packet_in packet, out headers_t hdr, inout metadata_t meta, inout standard_metadata_t standard_metadata) {
    state start { transition parse_ethernet; }
    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) { TYPE_IPV4: parse_ipv4; default: accept; }
    }
    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) { IP_PROTO_HWIMO: parse_hwimo; default: accept; }
    }
    state parse_hwimo { packet.extract(hdr.hwimo); transition parse_features; }
    state parse_features { packet.extract(hdr.features); transition parse_prediction; }
    state parse_prediction { packet.extract(hdr.prediction); transition parse_tcp; }
    state parse_tcp { packet.extract(hdr.tcp); transition accept; }
}

control MyVerifyChecksum(inout headers_t hdr, inout metadata_t meta) { apply { } }

control MyIngress(inout headers_t hdr, inout metadata_t meta, inout standard_metadata_t standard_metadata) {

    action set_pred_1(bit<1> v1) {
        hdr.prediction.pred_task1 = v1;
    }

    action set_pred_2(bit<1> v2) {
        hdr.prediction.pred_task2 = v2;
    }

    action set_pred_3(bit<3> v3) {
        hdr.prediction.pred_task3 = v3;
    }

    action set_pred_4(bit<3> v4) {
        hdr.prediction.pred_task4 = v4;
    }

    action set_pred_5_6_7(bit<2> v5, bit<2> v6, bit<2> v7) {
        hdr.prediction.pred_task5 = v5;
        hdr.prediction.pred_task6 = v6;
        hdr.prediction.pred_task7 = v7;
    }


    table tb_hwimo_tree_1 {
        key = {
            meta.key_1: ternary;
        }
        actions = {
            set_pred_1;
            NoAction;
        }
        size = 50;
        default_action = NoAction();
    }

    table tb_hwimo_tree_2 {
        key = {
            meta.key_2: ternary;
        }
        actions = {
            set_pred_2;
            NoAction;
        }
        size = 56;
        default_action = NoAction();
    }

    table tb_hwimo_tree_3 {
        key = {
            meta.key_3: ternary;
        }
        actions = {
            set_pred_3;
            NoAction;
        }
        size = 282;
        default_action = NoAction();
    }

    table tb_hwimo_tree_4 {
        key = {
            meta.key_4: ternary;
        }
        actions = {
            set_pred_4;
            NoAction;
        }
        size = 380;
        default_action = NoAction();
    }

    table tb_hwimo_tree_5_6_7 {
        key = {
            meta.key_5_6_7: ternary;
        }
        actions = {
            set_pred_5_6_7;
            NoAction;
        }
        size = 1468;
        default_action = NoAction();
    }

    apply {
        if (hdr.features.isValid()) {
        meta.key_1 = 0;
        meta.key_1[29:29] = hdr.features.bits[124:124];
        meta.key_1[28:28] = hdr.features.bits[122:122];
        meta.key_1[27:27] = hdr.features.bits[121:121];
        meta.key_1[26:26] = hdr.features.bits[119:119];
        meta.key_1[25:25] = hdr.features.bits[117:117];
        meta.key_1[24:24] = hdr.features.bits[116:116];
        meta.key_1[23:23] = hdr.features.bits[112:112];
        meta.key_1[22:22] = hdr.features.bits[111:111];
        meta.key_1[21:21] = hdr.features.bits[109:109];
        meta.key_1[20:20] = hdr.features.bits[108:108];
        meta.key_1[19:19] = hdr.features.bits[107:107];
        meta.key_1[18:18] = hdr.features.bits[103:103];
        meta.key_1[17:17] = hdr.features.bits[100:100];
        meta.key_1[16:16] = hdr.features.bits[98:98];
        meta.key_1[15:15] = hdr.features.bits[97:97];
        meta.key_1[14:14] = hdr.features.bits[92:92];
        meta.key_1[13:13] = hdr.features.bits[91:91];
        meta.key_1[12:12] = hdr.features.bits[87:87];
        meta.key_1[11:11] = hdr.features.bits[86:86];
        meta.key_1[10:10] = hdr.features.bits[77:77];
        meta.key_1[9:9] = hdr.features.bits[71:71];
        meta.key_1[8:8] = hdr.features.bits[66:66];
        meta.key_1[7:7] = hdr.features.bits[56:56];
        meta.key_1[6:6] = hdr.features.bits[53:53];
        meta.key_1[5:5] = hdr.features.bits[42:42];
        meta.key_1[4:4] = hdr.features.bits[39:39];
        meta.key_1[3:3] = hdr.features.bits[17:17];
        meta.key_1[2:2] = hdr.features.bits[14:14];
        meta.key_1[1:1] = hdr.features.bits[10:10];
        meta.key_1[0:0] = hdr.features.bits[0:0];
        tb_hwimo_tree_1.apply();
        meta.key_2 = 0;
        meta.key_2[29:29] = hdr.features.bits[141:141];
        meta.key_2[28:28] = hdr.features.bits[130:130];
        meta.key_2[27:27] = hdr.features.bits[128:128];
        meta.key_2[26:26] = hdr.features.bits[125:125];
        meta.key_2[25:25] = hdr.features.bits[124:124];
        meta.key_2[24:24] = hdr.features.bits[123:123];
        meta.key_2[23:23] = hdr.features.bits[119:119];
        meta.key_2[22:22] = hdr.features.bits[108:108];
        meta.key_2[21:21] = hdr.features.bits[106:106];
        meta.key_2[20:20] = hdr.features.bits[99:99];
        meta.key_2[19:19] = hdr.features.bits[98:98];
        meta.key_2[18:18] = hdr.features.bits[97:97];
        meta.key_2[17:17] = hdr.features.bits[89:89];
        meta.key_2[16:16] = hdr.features.bits[87:87];
        meta.key_2[15:15] = hdr.features.bits[82:82];
        meta.key_2[14:14] = hdr.features.bits[80:80];
        meta.key_2[13:13] = hdr.features.bits[78:78];
        meta.key_2[12:12] = hdr.features.bits[77:77];
        meta.key_2[11:11] = hdr.features.bits[71:71];
        meta.key_2[10:10] = hdr.features.bits[68:68];
        meta.key_2[9:9] = hdr.features.bits[67:67];
        meta.key_2[8:8] = hdr.features.bits[56:56];
        meta.key_2[7:7] = hdr.features.bits[55:55];
        meta.key_2[6:6] = hdr.features.bits[51:51];
        meta.key_2[5:5] = hdr.features.bits[46:46];
        meta.key_2[4:4] = hdr.features.bits[23:23];
        meta.key_2[3:3] = hdr.features.bits[20:20];
        meta.key_2[2:2] = hdr.features.bits[17:17];
        meta.key_2[1:1] = hdr.features.bits[15:15];
        meta.key_2[0:0] = hdr.features.bits[4:4];
        tb_hwimo_tree_2.apply();
        meta.key_3 = 0;
        meta.key_3[29:29] = hdr.features.bits[129:129];
        meta.key_3[28:28] = hdr.features.bits[120:120];
        meta.key_3[27:27] = hdr.features.bits[119:119];
        meta.key_3[26:26] = hdr.features.bits[117:117];
        meta.key_3[25:25] = hdr.features.bits[115:115];
        meta.key_3[24:24] = hdr.features.bits[114:114];
        meta.key_3[23:23] = hdr.features.bits[112:112];
        meta.key_3[22:22] = hdr.features.bits[109:109];
        meta.key_3[21:21] = hdr.features.bits[101:101];
        meta.key_3[20:20] = hdr.features.bits[98:98];
        meta.key_3[19:19] = hdr.features.bits[97:97];
        meta.key_3[18:18] = hdr.features.bits[95:95];
        meta.key_3[17:17] = hdr.features.bits[85:85];
        meta.key_3[16:16] = hdr.features.bits[84:84];
        meta.key_3[15:15] = hdr.features.bits[83:83];
        meta.key_3[14:14] = hdr.features.bits[80:80];
        meta.key_3[13:13] = hdr.features.bits[77:77];
        meta.key_3[12:12] = hdr.features.bits[72:72];
        meta.key_3[11:11] = hdr.features.bits[71:71];
        meta.key_3[10:10] = hdr.features.bits[69:69];
        meta.key_3[9:9] = hdr.features.bits[68:68];
        meta.key_3[8:8] = hdr.features.bits[67:67];
        meta.key_3[7:7] = hdr.features.bits[66:66];
        meta.key_3[6:6] = hdr.features.bits[65:65];
        meta.key_3[5:5] = hdr.features.bits[52:52];
        meta.key_3[4:4] = hdr.features.bits[49:49];
        meta.key_3[3:3] = hdr.features.bits[38:38];
        meta.key_3[2:2] = hdr.features.bits[25:25];
        meta.key_3[1:1] = hdr.features.bits[23:23];
        meta.key_3[0:0] = hdr.features.bits[14:14];
        tb_hwimo_tree_3.apply();
        meta.key_4 = 0;
        meta.key_4[29:29] = hdr.features.bits[130:130];
        meta.key_4[28:28] = hdr.features.bits[127:127];
        meta.key_4[27:27] = hdr.features.bits[125:125];
        meta.key_4[26:26] = hdr.features.bits[120:120];
        meta.key_4[25:25] = hdr.features.bits[118:118];
        meta.key_4[24:24] = hdr.features.bits[117:117];
        meta.key_4[23:23] = hdr.features.bits[115:115];
        meta.key_4[22:22] = hdr.features.bits[108:108];
        meta.key_4[21:21] = hdr.features.bits[106:106];
        meta.key_4[20:20] = hdr.features.bits[104:104];
        meta.key_4[19:19] = hdr.features.bits[103:103];
        meta.key_4[18:18] = hdr.features.bits[101:101];
        meta.key_4[17:17] = hdr.features.bits[99:99];
        meta.key_4[16:16] = hdr.features.bits[95:95];
        meta.key_4[15:15] = hdr.features.bits[93:93];
        meta.key_4[14:14] = hdr.features.bits[92:92];
        meta.key_4[13:13] = hdr.features.bits[89:89];
        meta.key_4[12:12] = hdr.features.bits[83:83];
        meta.key_4[11:11] = hdr.features.bits[76:76];
        meta.key_4[10:10] = hdr.features.bits[69:69];
        meta.key_4[9:9] = hdr.features.bits[68:68];
        meta.key_4[8:8] = hdr.features.bits[59:59];
        meta.key_4[7:7] = hdr.features.bits[54:54];
        meta.key_4[6:6] = hdr.features.bits[52:52];
        meta.key_4[5:5] = hdr.features.bits[47:47];
        meta.key_4[4:4] = hdr.features.bits[43:43];
        meta.key_4[3:3] = hdr.features.bits[39:39];
        meta.key_4[2:2] = hdr.features.bits[38:38];
        meta.key_4[1:1] = hdr.features.bits[33:33];
        meta.key_4[0:0] = hdr.features.bits[25:25];
        tb_hwimo_tree_4.apply();
        meta.key_5_6_7 = 0;
        meta.key_5_6_7[29:29] = hdr.features.bits[147:147];
        meta.key_5_6_7[28:28] = hdr.features.bits[129:129];
        meta.key_5_6_7[27:27] = hdr.features.bits[127:127];
        meta.key_5_6_7[26:26] = hdr.features.bits[123:123];
        meta.key_5_6_7[25:25] = hdr.features.bits[122:122];
        meta.key_5_6_7[24:24] = hdr.features.bits[121:121];
        meta.key_5_6_7[23:23] = hdr.features.bits[119:119];
        meta.key_5_6_7[22:22] = hdr.features.bits[117:117];
        meta.key_5_6_7[21:21] = hdr.features.bits[116:116];
        meta.key_5_6_7[20:20] = hdr.features.bits[104:104];
        meta.key_5_6_7[19:19] = hdr.features.bits[100:100];
        meta.key_5_6_7[18:18] = hdr.features.bits[97:97];
        meta.key_5_6_7[17:17] = hdr.features.bits[85:85];
        meta.key_5_6_7[16:16] = hdr.features.bits[83:83];
        meta.key_5_6_7[15:15] = hdr.features.bits[81:81];
        meta.key_5_6_7[14:14] = hdr.features.bits[72:72];
        meta.key_5_6_7[13:13] = hdr.features.bits[68:68];
        meta.key_5_6_7[12:12] = hdr.features.bits[65:65];
        meta.key_5_6_7[11:11] = hdr.features.bits[59:59];
        meta.key_5_6_7[10:10] = hdr.features.bits[55:55];
        meta.key_5_6_7[9:9] = hdr.features.bits[54:54];
        meta.key_5_6_7[8:8] = hdr.features.bits[45:45];
        meta.key_5_6_7[7:7] = hdr.features.bits[43:43];
        meta.key_5_6_7[6:6] = hdr.features.bits[41:41];
        meta.key_5_6_7[5:5] = hdr.features.bits[39:39];
        meta.key_5_6_7[4:4] = hdr.features.bits[38:38];
        meta.key_5_6_7[3:3] = hdr.features.bits[33:33];
        meta.key_5_6_7[2:2] = hdr.features.bits[6:6];
        meta.key_5_6_7[1:1] = hdr.features.bits[3:3];
        meta.key_5_6_7[0:0] = hdr.features.bits[2:2];
        tb_hwimo_tree_5_6_7.apply();
        }
        if (standard_metadata.ingress_port == 0) {
            standard_metadata.egress_spec = 1;
        } else {
            standard_metadata.egress_spec = 0;
        }
    }
}

control MyEgress(inout headers_t hdr, inout metadata_t meta, inout standard_metadata_t standard_metadata) { apply { } }
control MyComputeChecksum(inout headers_t hdr, inout metadata_t meta) { apply { } }

control MyDeparser(packet_out packet, in headers_t hdr) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.hwimo);
        packet.emit(hdr.features);
        packet.emit(hdr.prediction);
        packet.emit(hdr.tcp);
    }
}

V1Switch(MyParser(), MyVerifyChecksum(), MyIngress(), MyEgress(), MyComputeChecksum(), MyDeparser()) main;
