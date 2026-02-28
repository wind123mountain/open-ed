import json
from datasets import load_dataset

def extract_key_info(doc):
    """
    Trích xuất các trường: type, trigger_word, arguments (text, role) 
    từ một document và format lại để dễ so sánh.
    """
    info_list = []
    
    # Duyệt qua các event
    for event in doc.get('events', []):
        event_type = event.get('type')
        
        # Duyệt qua các mention trong event
        for mention in event.get('mention', []):
            trigger_word = mention.get('trigger_word')
            
            # Trích xuất list arguments (chỉ lấy text và role)
            args = []
            for arg in mention.get('arguments', []):
                args.append({
                    'text': arg.get('text'),
                    'role': arg.get('role')
                })
            
            # Sắp xếp arguments theo text để đảm bảo thứ tự không làm sai kết quả so sánh
            args = sorted(args, key=lambda x: (x['text'] or "", x['role'] or ""))
            
            info_list.append({
                'type': event_type,
                'trigger_word': trigger_word,
                'arguments': args
            })
            
    # Sắp xếp toàn bộ danh sách event theo type và trigger_word
    info_list = sorted(info_list, key=lambda x: (x['type'] or "", x['trigger_word'] or ""))
    return info_list

def compare_json_lists(list1, list2):
    """
    So sánh 2 list dữ liệu dựa trên document ID chung.
    """
    # Chuyển list thành dictionary với key là document id để tra cứu nhanh (O(1))
    dict1 = {doc['id']: doc for doc in list1}
    dict2 = {doc['id']: doc for doc in list2}
    
    # Tìm các document ID có mặt ở cả 2 list
    common_ids = set(dict1.keys()).intersection(set(dict2.keys()))
    
    if not common_ids:
        print("Không tìm thấy Document ID nào chung giữa 2 list!")
        return

    print("total: ", len(dict1), len(dict2))
    print(f"Tìm thấy {len(common_ids)} Document ID chung. Đang tiến hành so sánh...\n")
    print("-" * 50)
    
    failed_count = 0
    for doc_id in common_ids:
        # Trích xuất thông tin trọng tâm từ 2 list
        info1 = extract_key_info(dict1[doc_id])
        info2 = extract_key_info(dict2[doc_id])
        
        # So sánh trực tiếp 2 cấu trúc đã được chuẩn hóa
        if info1 == info2:
            # print(f"✅ Document ID: {doc_id} -> GIỐNG NHAU HOÀN TOÀN")
            pass
        else:
            print(f"❌ Document ID: {doc_id} -> CÓ SỰ KHÁC BIỆT")
            print("  🔻 Ở List 1:")
            print(f"    {json.dumps(info1, indent=2, ensure_ascii=False)}")
            print("  🔻 Ở List 2:")
            print(f"    {json.dumps(info2, indent=2, ensure_ascii=False)}")
            failed_count += 1 
        # print("-" * 50)

    print("failed_count: ", failed_count)


data = load_dataset("datht/geneva-event-dataset")
data_gen = load_dataset("datht/geneva-short-generated-dataset")
compare_json_lists(data['validation'], data_gen['validation'])