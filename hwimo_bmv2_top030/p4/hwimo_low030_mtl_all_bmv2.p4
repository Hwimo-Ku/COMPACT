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
    bit<30> key_1_2_3_4_5_6_7;
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

    action set_pred_1_2_3_4_5_6_7(bit<1> v1, bit<1> v2, bit<3> v3, bit<3> v4, bit<2> v5, bit<2> v6, bit<2> v7) {
        hdr.prediction.pred_task1 = v1;
        hdr.prediction.pred_task2 = v2;
        hdr.prediction.pred_task3 = v3;
        hdr.prediction.pred_task4 = v4;
        hdr.prediction.pred_task5 = v5;
        hdr.prediction.pred_task6 = v6;
        hdr.prediction.pred_task7 = v7;
    }


    table tb_hwimo_tree_1_2_3_4_5_6_7 {
        key = {
            meta.key_1_2_3_4_5_6_7: ternary;
        }
        actions = {
            set_pred_1_2_3_4_5_6_7;
            NoAction;
        }
        size = 875;
        default_action = NoAction();
    }

    apply {
        if (hdr.features.isValid()) {
        meta.key_1_2_3_4_5_6_7 = 0;
        meta.key_1_2_3_4_5_6_7[29:29] = hdr.features.bits[152:152];
        meta.key_1_2_3_4_5_6_7[28:28] = hdr.features.bits[151:151];
        meta.key_1_2_3_4_5_6_7[27:27] = hdr.features.bits[150:150];
        meta.key_1_2_3_4_5_6_7[26:26] = hdr.features.bits[149:149];
        meta.key_1_2_3_4_5_6_7[25:25] = hdr.features.bits[148:148];
        meta.key_1_2_3_4_5_6_7[24:24] = hdr.features.bits[146:146];
        meta.key_1_2_3_4_5_6_7[23:23] = hdr.features.bits[144:144];
        meta.key_1_2_3_4_5_6_7[22:22] = hdr.features.bits[142:142];
        meta.key_1_2_3_4_5_6_7[21:21] = hdr.features.bits[139:139];
        meta.key_1_2_3_4_5_6_7[20:20] = hdr.features.bits[138:138];
        meta.key_1_2_3_4_5_6_7[19:19] = hdr.features.bits[137:137];
        meta.key_1_2_3_4_5_6_7[18:18] = hdr.features.bits[136:136];
        meta.key_1_2_3_4_5_6_7[17:17] = hdr.features.bits[135:135];
        meta.key_1_2_3_4_5_6_7[16:16] = hdr.features.bits[134:134];
        meta.key_1_2_3_4_5_6_7[15:15] = hdr.features.bits[132:132];
        meta.key_1_2_3_4_5_6_7[14:14] = hdr.features.bits[131:131];
        meta.key_1_2_3_4_5_6_7[13:13] = hdr.features.bits[126:126];
        meta.key_1_2_3_4_5_6_7[12:12] = hdr.features.bits[96:96];
        meta.key_1_2_3_4_5_6_7[11:11] = hdr.features.bits[32:32];
        meta.key_1_2_3_4_5_6_7[10:10] = hdr.features.bits[31:31];
        meta.key_1_2_3_4_5_6_7[9:9] = hdr.features.bits[29:29];
        meta.key_1_2_3_4_5_6_7[8:8] = hdr.features.bits[28:28];
        meta.key_1_2_3_4_5_6_7[7:7] = hdr.features.bits[27:27];
        meta.key_1_2_3_4_5_6_7[6:6] = hdr.features.bits[21:21];
        meta.key_1_2_3_4_5_6_7[5:5] = hdr.features.bits[20:20];
        meta.key_1_2_3_4_5_6_7[4:4] = hdr.features.bits[19:19];
        meta.key_1_2_3_4_5_6_7[3:3] = hdr.features.bits[18:18];
        meta.key_1_2_3_4_5_6_7[2:2] = hdr.features.bits[16:16];
        meta.key_1_2_3_4_5_6_7[1:1] = hdr.features.bits[13:13];
        meta.key_1_2_3_4_5_6_7[0:0] = hdr.features.bits[9:9];
        tb_hwimo_tree_1_2_3_4_5_6_7.apply();
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
