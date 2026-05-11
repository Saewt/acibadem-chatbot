import json
import sys
import os

# Optionally use the sanitizer if it's available
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.sanitizer import fix_encoding_and_whitespace
except ImportError:
    def fix_encoding_and_whitespace(text):
        return text.strip() if isinstance(text, str) else text

def build_staff_map(input_file):
    staff_map = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("record_type") == "academic_staff_member":
                    payload = record.get("payload", {})
                    
                    unit = payload.get("unit_name", "Unknown Unit")
                    unit = fix_encoding_and_whitespace(unit)
                    
                    if unit not in staff_map:
                        staff_map[unit] = {
                            "unit_kind": payload.get("unit_kind"),
                            "parent_unit": fix_encoding_and_whitespace(payload.get("parent_unit_name")),
                            "faculty_members": []
                        }
                        
                    staff_member = {
                        "name": fix_encoding_and_whitespace(payload.get("staff_name")),
                        "title": fix_encoding_and_whitespace(payload.get("staff_title")),
                        "role": fix_encoding_and_whitespace(payload.get("staff_role"))
                    }
                    staff_map[unit]["faculty_members"].append(staff_member)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}", file=sys.stderr)
    return staff_map

def main():
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Fallback to the likely file since records_clean.jsonl contains the staff members
        input_file = 'data/acibadem-dataset/acibadem_output/records_clean.jsonl'
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}. Please provide the file path as an argument.")
            sys.exit(1)
            
    output_file = 'staff_map_debug.json'
    
    print(f"Analyzing academic_staff_member records in {input_file}...")
    staff_map = build_staff_map(input_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(staff_map, f, ensure_ascii=False, indent=2)
        
    total_staff = sum(len(dept["faculty_members"]) for dept in staff_map.values())
    print(f"Mapped {total_staff} staff members across {len(staff_map)} departments.")
    print(f"Output saved to {output_file}")

if __name__ == '__main__':
    main()
